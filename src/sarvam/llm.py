"""Sarvam LLM implementation for LangChain."""
import os
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from pydantic import Field, SecretStr

from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

from sarvam.sarvam_logging import logger


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

        # Override with any additional kwargs
        params.update(kwargs)

        # Log API call at DEBUG level
        logger.debug(
            f"Calling Sarvam API: model={self.model}, "
            f"prompt_length={len(prompt)}, "
            f"temperature={self.temperature}, "
            f"reasoning_effort={self.reasoning_effort}, "
            f"wiki_grounding={self.wiki_grounding}"
        )

        try:
            response = self._client.chat.completions(**params)
        except ApiError as e:
            logger.error(f"Sarvam API error: {e.body}")
            raise RuntimeError(f"Sarvam API error: {e.body}") from e

        # Log token usage at DEBUG level
        if hasattr(response, "usage"):
            logger.debug(
                f"Token usage: prompt={response.usage.prompt_tokens}, "
                f"completion={response.usage.completion_tokens}, "
                f"total={response.usage.total_tokens}"
            )

        return response.choices[0].message.content

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate from LLM."""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop, run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the Sarvam API response.

        Note: Sarvam API may not support streaming yet. This is a placeholder
        for future implementation when streaming becomes available.
        """
        # TODO: Implement streaming when Sarvam API supports it
        # For now, fall back to non-streaming
        text = self._call(prompt, stop, run_manager, **kwargs)
        yield GenerationChunk(text=text)

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
