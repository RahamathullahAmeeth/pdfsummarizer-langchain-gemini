# 📄 PDF AI Summarizer (LangChain + Gemini)

A Streamlit application that summarizes PDF documents with **Google Gemini** through **LangChain**. Upload a PDF, choose a summary style, and get a final summary, per-chunk insights, optional document analysis, key quotes, and downloadable reports.

## Features

- Extracts and cleans text from PDFs, then splits it into token-limited chunks.
- **Map → reduce** summarization: summarize each chunk, then merge into one cohesive summary.
- Four summary styles: `brief`, `detailed`, `bullet_points`, `executive`.
- Optional document-structure analysis and key-quote extraction.
- Export the final report as `.txt` and per-chunk statistics as `.csv`.
- Configurable model with a fallback when the primary Gemini model is unavailable.

## Project structure

```text
app.py            Streamlit UI (settings, upload, results, downloads)
pdf_processor.py  PDF text extraction, cleaning, token counting, chunking
summarizer.py     LangChain chains for summaries, analysis and quotes
prompts.py        All ChatPromptTemplates and style instructions
utils.py          Display helpers, exports, statistics, key validation
tests/            Unit tests for the processing and prompt layers
```

## Architecture

```text
PDF upload
   │
   ▼
extract text → clean → count tokens → chunk
   │                                  │
   │                                  ├─► chunk summaries (map, batched)
   │                                  │
   │                                  └─► combine summaries (reduce)
   ▼
metadata + analysis + quotes → Streamlit results → TXT / CSV downloads
```

## Setup (Windows / PowerShell)

Requires Python 3.10+ (tested on 3.15).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env and set GEMINI_API_KEY
```

Get a free API key at <https://aistudio.google.com/apikey>.

## Run

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal (usually <http://localhost:8501>).

## Configuration

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | yes | Google AI Studio API key |
| `GEMINI_MODEL` | no | Primary Gemini model (default `gemini-3.5-flash-lite`) |
| `GEMINI_FALLBACK_MODEL` | no | Model tried if the primary fails (default `gemini-flash-lite-latest`) |

The model can also be changed from the app sidebar.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Development phases

- **Phase 0** — validate Gemini key and GitHub login; initialize the private repo.
- **Phase 1** — scaffold files and install dependencies.
- **Phase 2** — PDF processing layer.
- **Phase 3** — LangChain/Gemini summarization layer.
- **Phase 4** — Streamlit UI.
- **Phase 5** — end-to-end verification.
- **Phase 6** — documentation, hardening and final push.

## Troubleshooting

- **No extractable text**: the PDF is likely scanned images; run OCR first.
- **429 / rate limits**: wait and retry, upload a smaller document, or pick another model in the sidebar.
- **404 / model not available**: set `GEMINI_MODEL` to a model your key can access, e.g. `gemini-3.5-flash-lite`.
- **API key not found**: ensure `.env` sits next to `app.py` and contains `GEMINI_API_KEY`.
