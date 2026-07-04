# Aurora AI Assistant (`aiModel`)

Amazon seller AI chatbot (campaigns, orders, inventory, profitability, restock).

It shares Aurora’s MongoDB `users` collection and JWT secret, so a user logged into Aurora can use the assistant without a second login.

## Run locally

```bash
# 1) Create aiModel/.env from auroraBackend/.env
python setup_env.py

# 2) Add your Groq key to aiModel/.env
# GROQ_API_KEY=gsk_...

# 3) Install Python deps (Python 3.11+ recommended)
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt

# Optional (Meta ads competitor tool)
# playwright install chromium

# 4) Start API + UI on port 8000
uvicorn main:app --reload --port 8000
```

## Use from Aurora

1. Start Aurora backend (`:5000`) and frontend (`:5173`).
2. Start this AI service (`:8000`).
3. Sign in to Aurora and open **AI Assistant** in the navbar (`/ai`).

Aurora embeds the assistant and passes your session token automatically.

Set `VITE_AI_URL` in `auroraFrontend/.env` if the AI service is not on `http://localhost:8000`.
