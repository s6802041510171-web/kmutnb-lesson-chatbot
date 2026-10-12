# KMUTNB Lesson Chatbot

The Vercel deployment serves the chat page and FastAPI backend from the same project. The browser sends chat requests to `/chat` on the current Vercel domain.

## Vercel setup

1. Import this GitHub repository into Vercel and deploy it with the Python runtime detected from `app.py` and `requirements.txt`.
2. In Vercel Project Settings, add the environment variable `TYPHOON_API_KEY` with the API key value. Apply it to Production and redeploy.
3. Open `/health` on the deployed domain. A successful response includes the number of indexed dataset records.

Keep API keys in Vercel Environment Variables or a local `.env` file. Never commit them to GitHub.

## Run locally

Install the dependencies from `requirements.txt`, set `TYPHOON_API_KEY` in `.env`, and start the API with `python app.py`. Open `http://localhost:10000` to use the chat page.
