"""Sarvam Chat Model implementation for LangChain."""
import asyncio
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import Field, SecretStr
from sarvamai.core import RequestOptions
from typing_extensions import override

from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

from sarvam.sarvam_logging import logger
from sarvam.chat.utils import extract_after_think

# Constants for feature limitations
_FEATURE_TOOLS_NOT_SUPPORTED = "tools_not_supported"


class SarvamChat(BaseChatModel):
    """Sarvam Chat Model wrapper for LangChain.

    Supports tool/function calling for sarvam-30b, sarvam-30b-16k, sarvam-105b, and sarvam-105b-32k models.
    sarvam-m does not support tools.

    Args:
        api_key: Sarvam API subscription key
        model: Model name to use (default: "sarvam-m"). Options: "sarvam-m", "sarvam-30b", "sarvam-30b-16k", "sarvam-105b", "sarvam-105b-32k"
        temperature: Sampling temperature (0-2)
        top_p: Nucleus sampling parameter
        reasoning_effort: Reasoning effort level ("low", "medium", "high")
        wiki_grounding: Enable wiki grounding for factual queries
        tool_choice: Tool choice mode (only for tool-supporting models): "none", "auto", "required", or specific tool
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
    wiki_grounding: bool = Field(default=False, description="Enable wiki grounding")
    bound_tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Tools bound to this model instance"
    )
    tool_choice: Optional[str] = Field(
        default=None,
        description="Tool choice mode: none, auto, required, or specific tool name. "
        "Only applicable for sarvam-30b and sarvam-105b models.",
    )
    max_retry: int = 3
    max_tokens: int = Field(default=8192, ge=1, description="Maximum tokens to generate")

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

    def _supports_tools(self) -> bool:
        """Check if the current model supports tool/function calling.

        Returns:
            True if the model supports tools (sarvam-30b, sarvam-105b and their variants), False otherwise.
        """
        return self.model in [
            "sarvam-30b",
            "sarvam-30b-16k",
            "sarvam-105b",
            "sarvam-105b-32k",
        ]

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
            elif isinstance(msg, ToolMessage):
                # ToolMessage requires tool_call_id and content
                converted.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
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

        # Set LangSmith metadata for tracing
        if run_manager:
            run_manager.metadata.update({
                "ls_provider": "sarvam",
                "ls_model_name": self.model,
            })

        # Build request parameters
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
        }

        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.wiki_grounding:
            params["wiki_grounding"] = True
        if self.max_retry:
            params["request_options"] = RequestOptions(
                max_retries=self.max_retry,
            )
        # Set max_tokens based on model context window, unless user explicitly set a different value
        # sarvam-m: 8192 tokens
        # sarvam-30b-16k: 8192 tokens (16k context total, need room for input)
        # All other models: 16384 tokens
        # If user passed a custom max_tokens, respect their choice
        if self.max_tokens != 8192:
            # User explicitly set a custom max_tokens value
            params["max_tokens"] = self.max_tokens
        elif self.model == "sarvam-30b-16k":
            params["max_tokens"] = 8192
        elif self.model == "sarvam-m":
            params["max_tokens"] = 8192
        else:
            params["max_tokens"] = 16384

        # Handle tool/function calling - model-specific support
        # sarvam-30b and sarvam-105b support tools, sarvam-m does not
        if self.bound_tools and self._supports_tools():
            # Pass tools to API for models that support it
            params["tools"] = self.bound_tools
            if self.tool_choice:
                params["tool_choice"] = self.tool_choice
            logger.debug(f"Tools passed to API for {self.model}: {[t['function']['name'] for t in self.bound_tools]}")
        elif self.bound_tools and not self._supports_tools():
            logger.info(
                f"Tool calling not supported for {self.model}. Tools will be ignored. "
                f"Use sarvam-30b or sarvam-105b for tool support."
            )
            logger.debug(f"Bound tools (not used): {[t['function']['name'] for t in self.bound_tools]}")

        # Override with any additional kwargs (tools already handled above)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "tools"}
        params.update(filtered_kwargs)

        # Log API call at DEBUG level
        logger.debug(
            f"Calling Sarvam API: model={self.model}, "
            f"messages={len(params['messages'])}, "
            f"temperature={self.temperature}, "
            f"reasoning_effort={self.reasoning_effort}, "
            f"wiki_grounding={self.wiki_grounding}, "
            f"max_tokens={self.max_tokens}, "
            f"max_retries={self.max_retry}"
        )

        # Application-level retry: Retry API call up to max_retry times on ApiError
        # Note: SDK also has internal retry via RequestOptions, providing additional resilience
        response = None
        for attempt in range(self.max_retry):
            try:
                response = self._client.chat.completions(**params)
                break  # Success - exit retry loop
            except ApiError as e:
                if attempt < self.max_retry - 1:
                    logger.warning(f"API error on attempt {attempt + 1}/{self.max_retry}: {e.body}. Retrying...")
                    continue  # Try again
                # Final attempt failed - notify LangSmith and raise
                if run_manager:
                    run_manager.on_llm_error(e)
                logger.error(f"Sarvam API error after {self.max_retry} attempts: {e.body}")
                raise RuntimeError(f"Sarvam API error: {e.body}") from e

        # Extract response
        message = response.choices[0].message
        raw_content = message.content
        content = extract_after_think(raw_content)

        # Handle tool calls in the response (for models that support tools)
        additional_kwargs = {}
        finish_reason = None
        if hasattr(message, "tool_calls") and message.tool_calls and isinstance(message.tool_calls, list):
            # Extract tool calls and store in additional_kwargs
            tool_calls_data = []
            for tc in message.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })
            additional_kwargs["tool_calls"] = tool_calls_data
            logger.debug(f"Tool calls received: {len(tool_calls_data)} calls")

        # Get finish reason if available
        if hasattr(response.choices[0], "finish_reason"):
            finish_reason = response.choices[0].finish_reason
            logger.debug(f"Finish reason: {finish_reason}")

        # Build token usage info if available (for LangSmith and response metadata)
        token_usage = None
        usage_metadata = None
        if hasattr(response, "usage") and response.usage:
            # Extract token counts, handling both real and mocked responses
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
            total_tokens = getattr(response.usage, "total_tokens", 0)

            # Only include if values are actual integers (not Mock objects)
            if isinstance(input_tokens, int) and isinstance(output_tokens, int) and isinstance(total_tokens, int):
                token_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
                # Also set usage_metadata for the AIMessage (LangChain 0.1+)
                usage_metadata = token_usage.copy()

        # Log token usage at DEBUG level
        if token_usage:
            logger.debug(
                f"Token usage: prompt={token_usage['input_tokens']}, "
                f"completion={token_usage['output_tokens']}, "
                f"total={token_usage['total_tokens']}"
            )

        # Create AIMessage with usage_metadata and additional_kwargs
        message_kwargs = {
            "content": content,
            **({"usage_metadata": usage_metadata} if usage_metadata else {}),
        }
        if additional_kwargs:
            message_kwargs["additional_kwargs"] = additional_kwargs

        generation = ChatGeneration(
            message=AIMessage(**message_kwargs),
            generation_info={
                "finish_reason": finish_reason,
                "token_usage": token_usage,
            }
        )

        return ChatResult(generations=[generation], llm_output={"token_usage": token_usage})

    def _stream(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the Sarvam API response.

        Note: Sarvam AI API does not support streaming yet. This implementation
        falls back to non-streaming and yields the complete response as a single chunk.
        When Sarvam AI adds streaming support, this can be updated to use native streaming.

        Args:
            messages: The input messages to send to the model
            stop: Optional list of stop strings
            run_manager: Optional callback manager for run tracking
            **kwargs: Additional arguments to pass to the model

        Yields:
            ChatGenerationChunk: A single chunk containing the complete response
        """
        if self._client is None:
            self._client = SarvamAI(api_subscription_key=self.api_key.get_secret_value())

        # Set LangSmith metadata for tracing
        if run_manager:
            run_manager.metadata.update({
                "ls_provider": "sarvam",
                "ls_model_name": self.model,
            })

        # Build request parameters (same as _generate)
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
        }

        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.wiki_grounding:
            params["wiki_grounding"] = True
        if self.max_retry:
            params["request_options"] = RequestOptions(
                max_retries=self.max_retry,
            )
        # Set max_tokens based on model context window, unless user explicitly set a different value
        # sarvam-m: 8192 tokens
        # sarvam-30b-16k: 8192 tokens (16k context total, need room for input)
        # All other models: 16384 tokens
        # If user passed a custom max_tokens, respect their choice
        if self.max_tokens != 8192:
            # User explicitly set a custom max_tokens value
            params["max_tokens"] = self.max_tokens
        elif self.model == "sarvam-30b-16k":
            params["max_tokens"] = 8192
        elif self.model == "sarvam-m":
            params["max_tokens"] = 8192
        else:
            params["max_tokens"] = 16384

        # Handle tool/function calling - model-specific support
        # sarvam-30b and sarvam-105b support tools, sarvam-m does not
        if self.bound_tools and self._supports_tools():
            # Pass tools to API for models that support it
            params["tools"] = self.bound_tools
            if self.tool_choice:
                params["tool_choice"] = self.tool_choice
            logger.debug(f"Tools passed to API for {self.model}: {[t['function']['name'] for t in self.bound_tools]}")
        elif self.bound_tools and not self._supports_tools():
            logger.info(
                f"Tool calling not supported for {self.model}. Tools will be ignored. "
                f"Use sarvam-30b or sarvam-105b for tool support."
            )
            logger.debug(f"Bound tools (not used): {[t['function']['name'] for t in self.bound_tools]}")

        # Override with any additional kwargs (tools already handled above)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "tools"}
        params.update(filtered_kwargs)

        # Log API call at DEBUG level
        logger.debug(
            f"Calling Sarvam API: model={self.model}, "
            f"messages={len(params['messages'])}, "
            f"temperature={self.temperature}, "
            f"reasoning_effort={self.reasoning_effort}, "
            f"wiki_grounding={self.wiki_grounding}, "
            f"max_tokens={self.max_tokens}, "
            f"max_retries={self.max_retry}"
        )

        # Application-level retry
        response = None
        for attempt in range(self.max_retry):
            try:
                response = self._client.chat.completions(**params)
                break
            except ApiError as e:
                if attempt < self.max_retry - 1:
                    logger.warning(f"API error on attempt {attempt + 1}/{self.max_retry}: {e.body}. Retrying...")
                    continue
                # Final attempt failed - notify LangSmith and raise
                if run_manager:
                    run_manager.on_llm_error(e)
                logger.error(f"Sarvam API error after {self.max_retry} attempts: {e.body}")
                raise RuntimeError(f"Sarvam API error: {e.body}") from e

        # Extract response
        message = response.choices[0].message
        raw_content = message.content
        content = extract_after_think(raw_content)

        # Handle tool calls in the response (for models that support tools)
        additional_kwargs = {}
        finish_reason = None
        if hasattr(message, "tool_calls") and message.tool_calls and isinstance(message.tool_calls, list):
            # Extract tool calls and store in additional_kwargs
            tool_calls_data = []
            for tc in message.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })
            additional_kwargs["tool_calls"] = tool_calls_data
            logger.debug(f"Tool calls received: {len(tool_calls_data)} calls")

        # Get finish reason if available
        if hasattr(response.choices[0], "finish_reason"):
            finish_reason = response.choices[0].finish_reason
            logger.debug(f"Finish reason: {finish_reason}")

        # Build token usage info if available
        token_usage = None
        usage_metadata = None
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
            total_tokens = getattr(response.usage, "total_tokens", 0)

            if isinstance(input_tokens, int) and isinstance(output_tokens, int) and isinstance(total_tokens, int):
                token_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
                usage_metadata = token_usage.copy()

        # Log token usage at DEBUG level
        if token_usage:
            logger.debug(
                f"Token usage: prompt={token_usage['input_tokens']}, "
                f"completion={token_usage['output_tokens']}, "
                f"total={token_usage['total_tokens']}"
            )

        # Create message kwargs for AIMessageChunk
        message_kwargs = {
            "content": content,
            **({"usage_metadata": usage_metadata} if usage_metadata else {}),
        }
        if additional_kwargs:
            message_kwargs["additional_kwargs"] = additional_kwargs

        # Yield as a single chunk
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(**message_kwargs),
            generation_info={
                "finish_reason": finish_reason,
                "token_usage": token_usage,
            }
        )

        yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "reasoning_effort": self.reasoning_effort,
            "wiki_grounding": self.wiki_grounding,
            "max_tokens": self.max_tokens,
        }

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, BaseTool, callable]],
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        """Bind tools to the chat model.

        Tool/function calling is supported for sarvam-30b and sarvam-105b models only.
        For sarvam-m, tools will be bound but not passed to the API.

        Args:
            tools: A list of tools to bind to the model. Can be:
                - Dictionaries following OpenAI function format
                - BaseTool instances
                - Python functions
            **kwargs: Additional arguments to pass (e.g., tool_choice for supported models)

        Returns:
            A new Runnable with tools bound

        Example:
            >>> from langchain_core.tools import tool
            >>> @tool
            >>> def get_weather(location: str) -> str:
            ...     return f"Sunny in {location}"
            >>> chat = SarvamChat(model="sarvam-30b")
            >>> bound_chat = chat.bind_tools([get_weather])
        """
        # Convert tools to Sarvam API format
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                # Check if it's already in the correct format (has "type" and "function")
                if "type" in tool and "function" in tool:
                    formatted_tools.append(tool)
                else:
                    # Wrap in the correct format
                    formatted_tools.append({
                        "type": "function",
                        "function": tool
                    })
            elif isinstance(tool, BaseTool):
                # Convert and wrap in Sarvam format
                function_def = convert_to_openai_function(tool)
                formatted_tools.append({
                    "type": "function",
                    "function": function_def
                })
            else:
                # Convert function and wrap in Sarvam format
                function_def = convert_to_openai_function(tool)
                formatted_tools.append({
                    "type": "function",
                    "function": function_def
                })

        # Create a new instance with tools bound
        bound_params = {
            **self._identifying_params,
            "api_key": self.api_key,
            "bound_tools": formatted_tools,
        }
        # Include tool_choice if it exists
        if self.tool_choice is not None:
            bound_params["tool_choice"] = self.tool_choice
        # Add any additional kwargs (they can override tool_choice)
        bound_params.update(kwargs)
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
        # Use the parent class's ainvoke which handles callbacks properly
        # The parent will call our _generate method with the correct run_manager
        return await super().ainvoke(input, config, stop=stop, **kwargs)
