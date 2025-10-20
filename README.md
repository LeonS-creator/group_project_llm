# GrantMatch Lite

GrantMatch Lite is a Gradio-based funding concierge that combines curated datasets, lightweight web scraping, and Mistral large language models to surface relevant grant and accelerator calls for students and early-stage startups in Sweden and the EU.

## Key capabilities
- Normalises funding pages from Vinnova, EU portals, and custom seed URLs into structured `CallRecord` entries by prompting the Mistral chat-completions API.
- Blends scraped information with local CSV records (`funding_website.csv`) and cached JSON (`data/calls_cache.json`) so restarts reuse previously discovered calls.
- Maintains a conversational startup profile (sector tags, stage, capital need, TRL, keywords, etc.) to personalise recommendations and follow-up questions.
- Scores opportunities deterministically before handing the top matches to the LLM, keeping responses grounded, cited, and tailored to the user profile.
- Switches prompts to deliver grant matchmaking, readiness coaching, or general funding advice depending on the user's intent.
- Ships with an Apple-inspired Gradio interface, quick commands (`refresh`, `profile`, `reset profile`, `ai only`, `addurl`), and profile progress feedback.

## Architecture overview
1. **Source discovery**: configurable scrapers (`SCRAPE_SOURCES`) collect candidate URLs from Vinnova, EU fund listings, and seed links (`SEED_URLS`).
2. **Content normalisation**: each page is cleaned with `readability-lxml`, then sent to the Mistral API (`normalize_call`) to extract deadlines, award ranges, TRL bands, must-haves, and descriptive summaries. Multiple prompts and fallbacks handle malformed JSON.
3. **Caching layer**: structured results are written to `calls_cache.json` so subsequent runs skip redundant normalisation. CSV inputs are merged on load to guarantee baseline coverage.
4. **Profile memory**: user chat turns update a `StartupProfile`, persisted in `data/profile.json` when `PROFILE_PERSIST=true`. The profile steers follow-up questions and scoring.
5. **Retrieval and ranking**: `filter_records` computes keyword, sector, TRL, geography, and ticket-size alignment scores, filtering out stale repeats or off-topic results.
6. **LLM response orchestration**: the top matches become a numbered context block. The `SYSTEM_PROMPT` instructs Mistral to recommend a single call with citations and next steps. Alternate prompts power startup coaching and general advice fallbacks.

## Local setup (Python)
1. Install Python 3.11 and dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file or export the required environment variables. At minimum set `MISTRAL_API_KEY`. Useful defaults:
   ```env
   MISTRAL_API_KEY=your_key
   MODEL_NAME=mistral-small-latest
   SCRAPE_SOURCES=seed,vinnova,eufund
   MAX_CALLS_PER_SOURCE=50
   SEED_URLS=https://eic.ec.europa.eu/eic-funding-opportunities/eic-accelerator_en
   CACHE_PATH=./data/calls_cache.json
   PROFILE_PERSIST=true
   PROFILE_PATH=./data/profile.json
   GRADIO_SHARE=false
   ```
3. Launch the app:
   ```bash
   python grant_chatbot.py
   ```
   Gradio serves the interface on http://127.0.0.1:7860 by default.

## Docker workflow
Build and run the container using the provided Dockerfile:
```bash
docker build -t grantmatch-lite .
docker run --rm -p 7860:7860 --env-file .env -v ${PWD}/data:/app/data grantmatch-lite
```

Or rely on Docker Compose for volume mounts and environment passthrough:
```bash
docker compose up --build
```
The service exposes port 7860 and mounts `funding_website.csv` plus the `data/` folder for cache persistence.

## Environment knobs
- `MISTRAL_API_KEY`: required for all LLM calls.
- `MODEL_NAME`: default `mistral-small-latest`; change to other Mistral deployments if needed.
- `SCRAPE_SOURCES`: comma-separated list of collectors to run (`seed`, `vinnova`, `eufund`).
- `SEED_URLS`: additional URLs that will always be normalised for demo reliability.
- `CACHE_PATH`: where structured call data is stored (`./data/calls_cache.json` by default).
- `PROFILE_PATH` and `PROFILE_PERSIST`: control profile storage between sessions.
- `GRADIO_SHARE`: set `true` to request a public Gradio share link (use with caution).
- `CSV_PATH` or `SHEET_CSV_URL`: optional extra datasets merged into the call catalog.
- `MAX_CALLS_PER_SOURCE`: throttle how many fresh pages each scraper processes per refresh.

## Chat commands and behaviour
- `refresh`: re-run the configured scrapers and update the cache.
- `profile`: print the current startup profile details.
- `reset profile` or `/reset`: clear stored profile data (confirmation required if populated).
- `ai only <question>`: filter recommendations to AI-relevant opportunities.
- `addurl https://...`: queue a custom opportunity page for one-off normalisation.

When the dataset is empty, the assistant falls back to a general funding prompt so the user still receives helpful guidance. Startup coaching prompts keep tone supportive while referencing the captured profile snapshot.

## Repository layout
```
grant_chatbot.py      # Scraping, Mistral orchestration, Gradio UI, chat logic
requirements.txt      # Python dependencies
Dockerfile            # Container build recipe
docker-compose.yml    # Local orchestration with mounted data and env passthrough
data/calls_cache.json # Cached opportunities (created at runtime)
funding_website.csv   # Seed dataset for instant recommendations
```

## Next steps
- Refresh sources after launch to populate the cache with current calls.
- Adjust scoring heuristics or prompts to better match target user groups.
- Add monitoring, API gateways, or rate limits before deploying beyond local use.
