# KMUTNB Lesson Chatbot

The Vercel deployment serves the chat page and FastAPI backend from this project. The backend searches `knowledge_base.md`, a readable Markdown index generated from the Question/Answer columns in `dataset.xlsx`, with the missing first programming step added from the user's source-document screenshot. It adds section headings and keywords to help match short questions. Answers are selected from indexed source text; the chatbot does not generate missing document content.

## Update the knowledge base

After changing `dataset.xlsx`, run `python build_knowledge_base.py` to regenerate `knowledge_base.md`. The script retains the screenshot-sourced first step because it is absent from the workbook. The backend uses the Markdown file when present and falls back to the Excel dataset if the file is missing or empty.

## Vercel setup

1. Import this GitHub repository into Vercel and deploy it with the Python runtime detected from `app.py` and `requirements.txt`.
2. Open `/health` on the deployed domain. A successful response includes the number of indexed records. No LLM API key is needed for document-only answers.

## Run locally

Install the dependencies from `requirements.txt`, start the API with `python app.py`, and open `http://localhost:10000` to use the chat page.
