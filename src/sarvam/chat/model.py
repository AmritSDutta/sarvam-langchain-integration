"""Sarvam Chat Model implementation for LangChain."""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import Field, SecretStr
from typing_extensions import override

from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

from sarvam.sarvam_logging import logger

# Constants for feature limitations
_FEATURE_TOOLS_NOT_SUPPORTED = "tools_not_supported"


class SarvamChat(BaseChatModel):
    """Sarvam Chat Model wrapper for LangChain.

    Args:
        api_key: Sarvam API subscription key
        model: Model name to use (default: "sarvam-m")
        temperature: Sampling temperature (0-2)
        top_p: Nucleus sampling parameter
        reasoning_effort: Reasoning effort level ("low", "medium", "high")
        wiki_grounding: Enable wiki grounding for factual queries
        **kwargs: Additional arguments
    """

    api_key: Optional[SecretStr] = Field(default=None, description="Sarvam API subscription key")
    model: str = Field(default="sarvam-m", description="Model name to use")
    temperature: float = Field(default=0.5, ge=0, le=2, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1, description="Nucleus sampling")
    reasoning_effort: Optional[str] = Field(
        default="high",
        description="Reasoning effort: low, medium, or high",
    )
    wiki_grounding: bool = Field(default=True, description="Enable wiki grounding")
    bound_tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Tools bound to this model instance"
    )

    _client: Optional[SarvamAI] = None

    def __init__(self, **kwargs: Any):
        # Extract api_key from kwargs or environment
        api_key_value = kwargs.pop("api_key", None) or os.environ.get("SARVAM_API_KEY")
        if api_key_value is None:
            raise ValueError(
                "Sarvam API key must be provided either via 'api_key' parameter or "
                "SARVAM_API_KEY environment variable"
            )
        if isinstance(api_key_value, str):
            api_key_value = SecretStr(api_key_value)
        kwargs["api_key"] = api_key_value

        super().__init__(**kwargs)
        self._client = SarvamAI(api_subscription_key=self.api_key.get_secret_value())

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "sarvam-chat"

    def _convert_messages(self, messages: Union[List[BaseMessage], str]) -> List[Dict[str, str]]:
        """Convert LangChain messages or string to Sarvam format."""
        # Handle string input - wrap in user message
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        converted = []
        for msg in messages:
            # Handle string items in list
            if isinstance(msg, str):
                converted.append({"role": "user", "content": msg})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            else:
                # Fallback for unknown message types
                content = msg if isinstance(msg, str) else getattr(msg, "content", str(msg))
                converted.append({"role": "user", "content": str(content)})
        return converted

    def _generate(
        self,
        messages: Union[List[BaseMessage], str],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response using Sarvam API."""
        if self._client is None:
            self._client = SarvamAI(api_subscription_key=self.api_key.get_secret_value())

        # Build request parameters
        params: Dict[str, Any] = {
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
        }

        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.wiki_grounding:
            params["wiki_grounding"] = True

        # Note: Sarvam API does not support tools/function calling yet
        # Tools are stored in bound_tools but not passed to API
        # This is for future compatibility when Sarvam adds support
        if self.bound_tools:
            logger.info(
                "Sarvam AI does not support tool/function calling yet. "
                "Tools are bound but will not be used by the API. "
                "This feature is for future compatibility."
            )
            logger.debug(f"Bound tools (not used): {[t.get('name') for t in self.bound_tools]}")

        # Override with any additional kwargs (but exclude tools)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "tools"}
        params.update(filtered_kwargs)

        # Log API call at DEBUG level
        logger.debug(
            f"Calling Sarvam API: model={self.model}, "
            f"messages={len(params['messages'])}, "
            f"temperature={self.temperature}, "
            f"reasoning_effort={self.reasoning_effort}, "
            f"wiki_grounding={self.wiki_grounding}"
        )

        try:
            response = self._client.chat.completions(**params)
        except ApiError as e:
            logger.error(f"Sarvam API error: {e.body}")
            raise RuntimeError(f"Sarvam API error: {e.body}") from e

        # Extract response
        message = response.choices[0].message
        content = message.content

        # Log token usage at DEBUG level
        if hasattr(response, "usage"):
            logger.debug(
                f"Token usage: prompt={response.usage.prompt_tokens}, "
                f"completion={response.usage.completion_tokens}, "
                f"total={response.usage.total_tokens}"
            )

        generation = ChatGeneration(message=AIMessage(content=content))

        # Build token usage info if available
        token_usage = None
        if hasattr(response, "usage"):
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ChatResult(generations=[generation], llm_output={"token_usage": token_usage})

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "reasoning_effort": self.reasoning_effort,
            "wiki_grounding": self.wiki_grounding,
        }

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, BaseTool, callable]],
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        """Bind tools to the chat model.

        Args:
            tools: A list of tools to bind to the model. Can be:
                - Dictionaries following OpenAI function format
                - BaseTool instances
                - Python functions
            **kwargs: Additional arguments to pass

        Returns:
            A new Runnable with tools bound
        """
        # Convert tools to OpenAI function format
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                formatted_tools.append(tool)
            elif isinstance(tool, BaseTool):
                formatted_tools.append(convert_to_openai_function(tool))
            else:
                formatted_tools.append(convert_to_openai_function(tool))

        # Create a new instance with tools bound
        bound_params = {
            **self._identifying_params,
            "api_key": self.api_key,
            "bound_tools": formatted_tools,
            **kwargs,
        }
        return self.__class__(**bound_params)

    @override
    async def ainvoke(
        self,
        input: Union[List[BaseMessage], str],
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Async invoke that runs the synchronous _generate in a thread pool.

        Args:
            input: The input messages to send to the model
            config: Optional configuration for the runnable
            stop: Optional list of stop strings
            **kwargs: Additional arguments to pass to the model

        Returns:
            The generated AI message

        Note:
            The Sarvam AI SDK does not support async operations natively.
            This method uses asyncio.to_thread to run the synchronous
            _generate method in a separate thread, preventing blocking
            of the event loop.
        """
        result = await asyncio.to_thread(
            self._generate,
            input,
            stop,
            None,  # run_manager
            **kwargs,
        )
        return result.generations[0].message
