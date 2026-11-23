"""Unit tests for prompt templates and style configuration."""

import unittest

from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES, STYLES


class PromptTests(unittest.TestCase):
    def test_all_four_styles_are_defined(self) -> None:
        self.assertEqual(
            set(STYLES), {"brief", "detailed", "bullet_points", "executive"}
        )

    def test_prompts_render_with_expected_variables(self) -> None:
        chunk = CHUNK_SUMMARY.format_messages(style="brief", text="Some document text")
        combined = COMBINE.format_messages(style="brief", summaries="Section 1: ...")
        analysis = ANALYSIS.format_messages(text="Some document text")
        quotes = QUOTES.format_messages(text="Some document text")

        self.assertEqual(len(chunk), 2)
        self.assertIn("Some document text", chunk[1].content)
        self.assertIn("Section 1: ...", combined[1].content)
        self.assertIn("Document Type", analysis[1].content)
        self.assertIn("verbatim", quotes[1].content)


if __name__ == "__main__":
    unittest.main()
