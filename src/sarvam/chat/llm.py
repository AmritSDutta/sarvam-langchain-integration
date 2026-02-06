"""Sarvam LLM implementation for LangChain."""
import asyncio
import os
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.runnables import RunnableConfig
from pydantic import Field, SecretStr
from sarvamai.core import RequestOptions
from typing_extensions import override

from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

from sarvam.sarvam_logging import logger
from sarvam.chat.utils import extract_after_think


class SarvamLLM(BaseLLM):
    """Sarvam LLM wrapper for LangChain.

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
        self._last_token_usage = None  # For LangSmith token usage tracking

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "sarvam"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Sarvam API."""
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
            "messages": [{"role": "user", "content": prompt}],
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
        params["max_tokens"] = self.max_tokens

        # Override with any additional kwargs
        params.update(kwargs)

        # Log API call at DEBUG level
        logger.debug(
            f"Calling Sarvam API: model={self.model}, "
            f"prompt_length={len(prompt)}, "
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

        # Extract token usage for LangSmith tracing
        token_usage = None
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
            total_tokens = getattr(response.usage, "total_tokens", 0)

            # Only include if values are actual integers
            if isinstance(input_tokens, int) and isinstance(output_tokens, int) and isinstance(total_tokens, int):
                token_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }

        # Log token usage at DEBUG level
        if token_usage:
            logger.debug(
                f"Token usage: prompt={token_usage['input_tokens']}, "
                f"completion={token_usage['output_tokens']}, "
                f"total={token_usage['total_tokens']}"
            )

        # Store token_usage as an instance variable for _generate to access
        self._last_token_usage = token_usage

        # Extract content after `` tag if present
        raw_content = response.choices[0].message.content
        content = extract_after_think(raw_content)

        return content

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate from LLM."""
        generations = []
        token_usage_sum = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        for prompt in prompts:
            text = self._call(prompt, stop, run_manager, **kwargs)
            generations.append([Generation(text=text)])

            # Accumulate token usage
            if hasattr(self, '_last_token_usage') and self._last_token_usage:
                token_usage_sum["input_tokens"] += self._last_token_usage["input_tokens"]
                token_usage_sum["output_tokens"] += self._last_token_usage["output_tokens"]
                token_usage_sum["total_tokens"] += self._last_token_usage["total_tokens"]

        return LLMResult(
            generations=generations,
            llm_output={"token_usage": token_usage_sum} if any(token_usage_sum.values()) else None
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the Sarvam API response.

        Note: Sarvam AI API does not support streaming yet. This implementation
        falls back to non-streaming and yields the complete response as a single chunk.
        When Sarvam AI adds streaming support, this can be updated to use native streaming.

        Args:
            prompt: The input prompt to send to the model
            stop: Optional list of stop strings
            run_manager: Optional callback manager for run tracking
            **kwargs: Additional arguments to pass to the model

        Yields:
            GenerationChunk: A single chunk containing the complete response
        """
        text = self._call(prompt, stop, run_manager, **kwargs)
        chunk = GenerationChunk(text=text)

        if run_manager:
            run_manager.on_llm_new_token(text, chunk=chunk)

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

    @override
    async def ainvoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Async invoke that runs the synchronous _call in a thread pool.

        Args:
            input: The input prompt to send to the model
            config: Optional configuration for the runnable
            stop: Optional list of stop strings
            **kwargs: Additional arguments to pass to the model

        Returns:
            The generated text response

        Note:
            The Sarvam AI SDK does not support async operations natively.
            This method uses asyncio.to_thread to run the synchronous
            _call method in a separate thread, preventing blocking
            of the event loop.
        """
        # Use the parent class's ainvoke which handles callbacks properly
        # The parent will call our _call method with the correct run_manager
        return await super().ainvoke(input, config, stop=stop, **kwargs)
