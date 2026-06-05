"""Integration tests for the MiniMax LLM provider.

These tests call the real MiniMax API and require the ``MINIMAX_API_KEY``
environment variable to be set.  They are skipped automatically when the
key is absent.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
SKIP_REASON = "MINIMAX_API_KEY not set"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMinimaxLLMIntegration(unittest.TestCase):
    """Live API integration tests."""

    def test_predict_m3(self):
        from tinygraph.llm.minimax import minimaxLLM

        llm = minimaxLLM(model_name="MiniMax-M3", api_key=MINIMAX_API_KEY)
        result = llm.predict("Say 'hello' and nothing else.")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_predict_m27(self):
        from tinygraph.llm.minimax import minimaxLLM

        llm = minimaxLLM(model_name="MiniMax-M2.7", api_key=MINIMAX_API_KEY)
        result = llm.predict("Say 'hello' and nothing else.")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_predict_m27_highspeed(self):
        from tinygraph.llm.minimax import minimaxLLM

        llm = minimaxLLM(
            model_name="MiniMax-M2.7-highspeed", api_key=MINIMAX_API_KEY
        )
        result = llm.predict("Say 'hi' and nothing else.")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_predict_returns_nonempty_content(self):
        from tinygraph.llm.minimax import minimaxLLM

        llm = minimaxLLM(api_key=MINIMAX_API_KEY)
        result = llm.predict("What is 1+1? Reply with just the number.")
        self.assertIsInstance(result, str)
        self.assertIn("2", result)


if __name__ == "__main__":
    unittest.main()
