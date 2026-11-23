# Walkthrough

## 1. Configure credentials

1. Create a Gemini API key at <https://aistudio.google.com/apikey>.
2. Copy `.env.example` to `.env`.
3. Set `GEMINI_API_KEY=...`. Optionally set `GEMINI_MODEL` and `GEMINI_FALLBACK_MODEL`.

> `.env` is git-ignored. Never commit your real key.

## 2. Install and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## 3. Use the app

1. Open the URL printed by Streamlit.
2. In the sidebar, pick a **Model**, **Summary Type**, chunk size, and optional analysis/quotes.
3. Upload a PDF. Metadata (pages, title, author, subject) appears immediately.
4. Click **Process PDF**.
5. Review the final summary, per-chunk summaries, analysis and quotes.
6. Download the report as text or the statistics as CSV.

## 4. How processing works

- `pdf_processor.py` extracts all page text, removes control characters, normalizes whitespace, and splits on sentence boundaries into chunks no larger than the chosen token limit.
- `summarizer.py` sends every chunk through a LangChain chain in parallel (map step), then merges the chunk summaries through a second chain (reduce step).
- If the configured model fails, a fallback model from `.env` is tried automatically.
- Results are stored in `st.session_state` so downloads and reruns do not re-process the PDF.

## 5. Adding a summary style

1. Add an entry to `STYLES` in `prompts.py`.
2. Add its key to the **Summary Type** selector in `app.py`.

## 6. Running tests

```powershell
python -m unittest discover -s tests -v
```
