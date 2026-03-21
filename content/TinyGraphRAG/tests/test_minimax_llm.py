"""Unit tests for the MiniMax LLM provider."""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Allow imports from the parent package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tinygraph.llm.minimax import minimaxLLM
from tinygraph.llm.base import BaseLLM


class TestMinimaxLLMInit(unittest.TestCase):
    """Tests for minimaxLLM initialisation."""

    @patch("tinygraph.llm.minimax.OpenAI")
    def test_is_base_llm_subclass(self, mock_openai):
        llm = minimaxLLM(api_key="test-key")
        self.assertIsInstance(llm, BaseLLM)

    @patch("tinygraph.llm.minimax.OpenAI")
    def test_default_model_name(self, mock_openai):
        llm = minimaxLLM(api_key="test-key")
        self.assertEqual(llm.model_name, "MiniMax-M2.7")

    @patch("tinygraph.llm.minimax.OpenAI")
    def test_custom_model_name(self, mock_openai):
        llm = minimaxLLM(model_name="MiniMax-M2.7-highspeed", api_key="test-key")
        self.assertEqual(llm.model_name, "MiniMax-M2.7-highspeed")

    @patch("tinygraph.llm.minimax.OpenAI")
    def test_default_base_url(self, mock_openai):
        minimaxLLM(api_key="test-key")
        mock_openai.assert_called_once_with(
            api_key="test-key", base_url="https://api.minimax.io/v1"
        )

    @patch("tinygraph.llm.minimax.OpenAI")
    def test_custom_base_url(self, mock_openai):
        minimaxLLM(api_key="test-key", base_url="https://api.minimaxi.com/v1")
        mock_openai.assert_called_once_with(
            api_key="test-key", base_url="https://api.minimaxi.com/v1"
        )

    @patch("tinygraph.llm.minimax.OpenAI")
    def test_model_params_default_empty(self, mock_openai):
        llm = minimaxLLM(api_key="test-key")
        self.assertEqual(llm.model_params, {})

    @patch("tinygraph.llm.minimax.OpenAI")
    def test_model_params_passed(self, mock_openai):
        params = {"temperature": 0.7}
        llm = minimaxLLM(api_key="test-key", model_params=params)
        self.assertEqual(llm.model_params, params)


class TestMinimaxLLMPredict(unittest.TestCase):
    """Tests for the predict method."""

    @patch("tinygraph.llm.minimax.OpenAI")
    def test_predict_returns_string(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from MiniMax!"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        llm = minimaxLLM(api_key="test-key")
        result = llm.predict("Hi")

        self.assertEqual(result, "Hello from MiniMax!")

    @patch("tinygraph.llm.minimax.OpenAI")
    def test_predict_sends_correct_messages(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        llm = minimaxLLM(model_name="MiniMax-M2.7", api_key="test-key")
        llm.predict("test input")

        mock_client.chat.completions.create.assert_called_once_with(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "test input"}],
        )

    @patch("tinygraph.llm.minimax.OpenAI")
    def test_predict_uses_highspeed_model(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "fast response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        llm = minimaxLLM(model_name="MiniMax-M2.7-highspeed", api_key="test-key")
        result = llm.predict("speed test")

        mock_client.chat.completions.create.assert_called_once_with(
            model="MiniMax-M2.7-highspeed",
            messages=[{"role": "user", "content": "speed test"}],
        )
        self.assertEqual(result, "fast response")

    @patch("tinygraph.llm.minimax.OpenAI")
    def test_predict_long_input(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        long_text = "word " * 1000
        mock_choice = MagicMock()
        mock_choice.message.content = "summary"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        llm = minimaxLLM(api_key="test-key")
        result = llm.predict(long_text)
        self.assertEqual(result, "summary")

    @patch("tinygraph.llm.minimax.OpenAI")
    def test_predict_empty_input(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        llm = minimaxLLM(api_key="test-key")
        result = llm.predict("")
        self.assertEqual(result, "")


class TestMinimaxLLMErrorHandling(unittest.TestCase):
    """Tests for error scenarios."""

    @patch("tinygraph.llm.minimax.OpenAI")
    def test_predict_propagates_api_error(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        llm = minimaxLLM(api_key="bad-key")
        with self.assertRaises(Exception) as ctx:
            llm.predict("test")
        self.assertIn("API error", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
