"""PDF text extraction, cleaning and token-aware chunking."""

import logging
import re

import tiktoken
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Turn an uploaded PDF into clean, token-limited text chunks."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Return the text of all pages; pages that fail to extract are skipped."""
        try:
            pdf_file.seek(0)
            reader = PdfReader(pdf_file)
            pages = []
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    pages.append(page.extract_text() or "")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping page %d: %s", page_num, exc)
            return "\n\n".join(pages)
        except Exception as exc:
            raise RuntimeError(f"Error reading PDF: {exc}") from exc

    def clean_text(self, text: str) -> str:
        """Drop control characters and normalize whitespace, keeping paragraphs."""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def count_tokens(self, text: str) -> int:
        """Approximate Gemini token count using tiktoken."""
        return len(self.encoding.encode(text, disallowed_special=()))

    def chunk_text(self, text: str, max_tokens: int = 8000) -> list[str]:
        """Split text into chunks of at most ``max_tokens`` on sentence boundaries."""
        if not text.strip():
            raise ValueError("No extractable text — the PDF may be scanned images")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        pieces: list[tuple[str, int]] = []
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            tokens = self.encoding.encode(sentence, disallowed_special=())
            if len(tokens) > max_tokens:  # hard-split oversized pieces by tokens
                pieces.extend(
                    (
                        self.encoding.decode(tokens[i : i + max_tokens]),
                        min(max_tokens, len(tokens) - i),
                    )
                    for i in range(0, len(tokens), max_tokens)
                )
            else:
                pieces.append((sentence, len(tokens)))

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for piece, token_count in pieces:
            if current and current_tokens + token_count + 1 > max_tokens:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            current.append(piece)
            current_tokens += token_count + 1  # +1 approximates the joining space
        if current:
            chunks.append(" ".join(current))
        return chunks

    def get_pdf_metadata(self, pdf_file) -> dict:
        """Return num_pages, title, author and subject (or {'error': ...})."""
        try:
            pdf_file.seek(0)
            reader = PdfReader(pdf_file)
            meta = reader.metadata or {}

            def field(key: str) -> str:
                return str(meta.get(key) or "Unknown")

            return {
                "num_pages": len(reader.pages),
                "title": field("/Title"),
                "author": field("/Author"),
                "subject": field("/Subject"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
