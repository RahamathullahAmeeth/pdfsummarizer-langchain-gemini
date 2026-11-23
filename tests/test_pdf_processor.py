"""Unit tests for PDF processing (no network or API key required)."""

import io
import unittest

from pypdf import PdfWriter

from pdf_processor import PDFProcessor


class PDFProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = PDFProcessor()

    def test_clean_text_normalizes_whitespace_and_control_chars(self) -> None:
        raw = "Hello\x00 world.   Multiple   spaces\n\n\nand lines."
        self.assertEqual(
            self.processor.clean_text(raw),
            "Hello world. Multiple spaces\n\nand lines.",
        )

    def test_chunk_text_respects_token_limit(self) -> None:
        text = " ".join(f"Sentence number {i} has several useful words." for i in range(80))
        chunks = self.processor.chunk_text(text, max_tokens=40)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(self.processor.count_tokens(chunk), 40)

    def test_chunk_text_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            self.processor.chunk_text("   ")

    def test_extract_and_metadata_from_blank_pdf(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.add_metadata({"/Title": "Sample", "/Author": "Tester"})
        buffer = io.BytesIO()
        writer.write(buffer)

        self.assertEqual(self.processor.extract_text_from_pdf(buffer), "")
        metadata = self.processor.get_pdf_metadata(buffer)
        self.assertEqual(metadata["num_pages"], 1)
        self.assertEqual(metadata["title"], "Sample")
        self.assertEqual(metadata["author"], "Tester")


if __name__ == "__main__":
    unittest.main()
