"""LangChain chains that call Gemini for summaries, analysis and quotes."""

import os
import re

from dotenv import load_dotenv
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES, STYLES

load_dotenv()

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_FALLBACK = "gemini-flash-lite-latest"
MAX_CONCURRENCY = 4


def _text_parser() -> RunnableLambda:
    """Gemini 3 returns list blocks; .text joins them into one string."""
    return RunnableLambda(lambda message: str(message.text))


def _build_chains(llm) -> dict[str, Runnable]:
    parser = _text_parser()
    return {
        "chunk": CHUNK_SUMMARY | llm | parser,
        "combine": COMBINE | llm | parser,
        "analysis": ANALYSIS | llm | parser,
        "quotes": QUOTES | llm | parser,
    }


class PDFSummarizer:
    """Map/reduce summarizer with an automatic fallback model."""

    def __init__(self, model: str | None = None, fallback_model: str | None = None):
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        fallback = fallback_model or os.getenv("GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK)
        self.fallback_model = fallback if fallback != self.model else None
        api_key = os.getenv("GEMINI_API_KEY")

        self.models = [self.model] + ([self.fallback_model] if self.fallback_model else [])
        self.chains_by_model = {
            name: _build_chains(
                ChatGoogleGenerativeAI(
                    model=name,
                    temperature=0.3,
                    google_api_key=api_key,
                )
            )
            for name in self.models
        }

    @staticmethod
    def _style(summary_type: str) -> str:
        return STYLES.get(summary_type, STYLES["detailed"])

    def _invoke(self, name: str, payload: dict) -> str:
        """Try the primary model, then the fallback, raising the last error."""
        last_error: Exception | None = None
        for model in self.models:
            try:
                return self.chains_by_model[model][name].invoke(payload)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise last_error or RuntimeError("No Gemini model available")

    def _batch(self, name: str, payloads: list[dict]) -> list[str | Exception]:
        """Batch with the primary model; retry the whole batch on fallback failure."""
        last_error: Exception | None = None
        for model in self.models:
            try:
                return self.chains_by_model[model][name].batch(
                    payloads,
                    config={"max_concurrency": MAX_CONCURRENCY},
                    return_exceptions=True,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise last_error or RuntimeError("No Gemini model available")

    def summarize_text(self, text: str, summary_type: str = "detailed") -> str:
        """Summarize a single piece of text."""
        return self._invoke("chunk", {"style": self._style(summary_type), "text": text})

    def summarize_chunks(self, chunks: list[str], summary_type: str = "detailed") -> dict:
        """Summarize each chunk (map) and merge the results (reduce)."""
        style = self._style(summary_type)
        payloads = [{"style": style, "text": chunk} for chunk in chunks]
        results = self._batch("chunk", payloads)

        summaries = []
        for index, (chunk, result) in enumerate(zip(chunks, results), start=1):
            ok = not isinstance(result, Exception)
            summary = result if ok else f"Error processing chunk: {result}"
            summaries.append(
                {
                    "chunk_number": index,
                    "summary": summary,
                    "ok": ok,
                    "original_length": len(chunk),
                    "summary_length": len(summary) if ok else 0,
                }
            )

        return {
            "individual_summaries": summaries,
            "combined_summary": self.combine_summaries(summaries, summary_type),
            "total_chunks": len(chunks),
            "summary_type": summary_type,
        }

    def combine_summaries(self, chunk_summaries: list[dict], summary_type: str) -> str:
        """Merge per-chunk summaries into one final summary."""
        valid = [item["summary"] for item in chunk_summaries if item.get("ok")]
        if not valid:
            return "No valid summaries were generated."
        if len(valid) == 1:
            return valid[0]

        sections = "\n\n".join(
            f"Section {index}: {summary}" for index, summary in enumerate(valid, start=1)
        )
        try:
            return self._invoke(
                "combine", {"style": self._style(summary_type), "summaries": sections}
            )
        except Exception:  # noqa: BLE001
            return f"Combined Summary:\n\n{sections}"

    def analyze_document_structure(self, text: str) -> dict:
        """Describe the document type, themes and purpose."""
        try:
            analysis = self._invoke("analysis", {"text": text[:3000]})
            return {"analysis": analysis, "status": "success"}
        except Exception as exc:  # noqa: BLE001
            return {"analysis": f"Error analyzing document: {exc}", "status": "error"}

    def extract_key_quotes(self, text: str) -> list[str]:
        """Extract up to 10 verbatim key quotes, one per line."""
        response = self._invoke("quotes", {"text": text})
        lines = (
            re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip(' "“”')
            for line in response.splitlines()
        )
        return [line for line in lines if line][:10]
