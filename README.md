# KMUTNB Lesson Chatbot

The Vercel deployment serves the chat page and FastAPI backend from the same project. The browser sends chat requests to `/chat` on the current Vercel domain. The backend reads the Question/Answer columns from `dataset.xlsx`, matches queries against source answers, and returns the selected Answer cell verbatim. If it cannot find a confident match, it says no matching document answer was found instead of generating one.

## Vercel setup

1. Import this GitHub repository into Vercel and deploy it with the Python runtime detected from `app.py` and `requirements.txt`.
2. Open `/health` on the deployed domain. A successful response includes the number of dataset records loaded from Excel. No LLM API key is needed for document-only answers.

## Run locally

Install the dependencies from `requirements.txt`, start the API with `python app.py`, and open `http://localhost:10000` to use the chat page.
