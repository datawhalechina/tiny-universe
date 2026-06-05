from openai import OpenAI
from typing import Any, Optional
from .base import BaseLLM


class minimaxLLM(BaseLLM):
    """Implementation of the BaseLLM interface using MiniMax's OpenAI-compatible API.

    MiniMax provides an OpenAI-compatible chat completions endpoint, so we reuse
    the ``openai`` SDK with a custom ``base_url``.

    Available models:
        - MiniMax-M3 (default, 512K context, up to 128K output, image input)
        - MiniMax-M2.7
        - MiniMax-M2.7-highspeed

    Args:
        model_name: Model identifier, e.g. ``"MiniMax-M3"``.
        api_key: MiniMax API key (``MINIMAX_API_KEY``).
        base_url: API base URL. Defaults to the global endpoint
            ``https://api.minimax.io/v1``.  Use
            ``https://api.minimaxi.com/v1`` for the China endpoint.
        model_params: Extra parameters forwarded to the chat completions call.
    """

    def __init__(
        self,
        model_name: str = "MiniMax-M3",
        api_key: str = "",
        base_url: str = "https://api.minimax.io/v1",
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, model_params, **kwargs)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def predict(self, input: str) -> str:
        """Sends a text input to the MiniMax model and retrieves a response.

        Args:
            input (str): Text sent to the MiniMax model

        Returns:
            str: The response from the MiniMax model.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": input}],
        )
        return response.choices[0].message.content
