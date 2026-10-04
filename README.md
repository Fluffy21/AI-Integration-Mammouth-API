# Mammouth Multi-Chat (Streamlit)

Multiple chats, each persisted as its own JSON file under `data/chats/`. The API key
stays server-side (Streamlit runs everything in the backend, the browser never sees it).

## Setup

```bash
cp .env.example .env
# add your API key to .env (see https://mammouth.ai/app/account/settings/api)
docker compose up -d --build
```

App: http://localhost:8501

## Without Docker

```bash
pip install -r requirements.txt
export MAMMOUTH_API_KEY=your_key
streamlit run app.py
```

## Features

- Multiple parallel chats, each persisted as JSON (`data/chats/<id>.json`)
- Name/Save/Delete per chat
- Model dropdown: provider, model, input/output cost per 1M tokens
  - Source: `https://api.mammouth.ai/public/models`, cached once per hour (`st.cache_data(ttl=3600)`),
    with a file fallback (`data/models_cache.json`) if the API is ever unreachable
- System prompt saved individually per chat, default "Caveman"
  (direct, correct, low token usage), custom presets stored in `data/prompts.json`
- File upload per chat (text/code files, PDF, DOCX, and images for vision models),
  kept as context for that chat and persisted under `data/files/<chat_id>/`
- Chat scroll: "Top"/"Bottom" buttons (best-effort via JS, Streamlit has no native scroll API)

## Data

Everything lives under `data/` (mounted as a volume) -> survives container restarts/updates.