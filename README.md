# AI Assessment Recommender

A conversational agent that helps hiring managers find the right SHL Individual Test Solutions through multi-turn dialogue.

## Architecture

```
POST /chat
    ↓
FastAPI Backend
    ↓
Semantic Retrieval Layer
(SentenceTransformers + FAISS)
    ↓
Groq LLM
(llama-3.3-70b-versatile)
    ↓
Catalog Validation Layer
(catalog.json grounding)
    ↓
Structured JSON Response
    ↕
Stateless Conversation History
```

**Key design decisions:**
- **Stateless API**: Full conversation history in every request. No session storage needed, trivial to scale.
- **Catalog-grounded**: Groq's system prompt includes the full catalog. Every recommendation is validated against `catalog.json` before returning — no hallucinated URLs possible.
- **JSON-only LLM output**: The agent is prompted to return only JSON, making structured extraction reliable without brittle regex parsing.
- **Turn cap enforcement**: Hard limit of 8 turns in Python code, not just prompting.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API key

```bash
export GROQ_API_KEY=your-key-here
```

### 3. Run locally

```bash
uvicorn main:app --reload --port 8000
```

### 4. Test

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a Java developer who works with stakeholders"}
    ]
  }'
```

## Catalog

`catalog.json` contains 34 SHL Individual Test Solutions (scraped and manually curated from the SHL product catalog). Each entry includes:

- `name` — exact product name
- `url` — product catalog URL
- `test_type` — A (Ability), B (Behaviour), K (Knowledge), P (Personality), S (Simulation)
- `description` — what the test measures
- `job_levels` — entry / graduate / professional / manager / executive / director
- `competencies` — skills and traits assessed
- `duration_minutes`, `remote_testing`, `adaptive`, `languages`

To refresh the catalog, run:
```bash
pip install requests beautifulsoup4 lxml
python scraper.py
```

## API Reference

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response:**
```json
{
  "reply": "...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` when clarifying or refusing
- `recommendations` has 1-10 items when committing to a shortlist
- `end_of_conversation: true` when the agent considers the task complete

## Evaluation

```bash
# Run against sample traces
python eval.py --url http://localhost:8000 --traces traces/

# Metrics computed:
# - Mean Recall@10 across all traces
# - Average turns to recommendation
# - Per-trace breakdown
```

## Deployment

### Render (free tier)

1. Push to GitHub
2. Connect repo in [Render dashboard](https://render.com)
3. Set `GROQ_API_KEY` in environment variables
4. Deploy — Render uses `render.yaml` automatically

### Docker

```bash
docker build -t shl-recommender .
docker run -e GROQ_API_KEY=your-key -p 8000:8000 shl-recommender
```

### Railway / Fly.io

Both support Python + `requirements.txt` autodeploy. Set the `GROQ_API_KEY` environment variable in the dashboard.

## Conversational Behaviors

| Behavior | Trigger | Action |
|----------|---------|--------|
| **Clarify** | Vague query ("I need an assessment") | Ask one focused question |
| **Recommend** | Enough context (role + level) | Return 1-10 shortlisted assessments |
| **Refine** | Mid-conversation constraint change | Update shortlist, preserve context |
| **Compare** | "Difference between X and Y?" | Ground answer in catalog data |
| **Refuse** | Off-topic, legal, competitor questions | Politely decline, redirect |

## Scope guardrails

- Only recommends assessments present in `catalog.json`
- URL validator strips any hallucinated URLs before responding
- Refuses: general hiring advice, legal questions, salary questions, prompt injection attempts
