# grant_chatbot.py
"""
GrantMatch Lite — a Mistral + scraping chatbot for student/startup funding calls.

Run:
  export MISTRAL_API_KEY=...     # or set in .env
  python grant_chatbot.py

Env knobs:
  MODEL_NAME=mistral-small-latest
  GRADIO_SHARE=false
  CACHE_PATH=./calls_cache.json
  SCRAPE_SOURCES=seed           # seed|vinnova|eufund (comma-separated)
  MAX_CALLS_PER_SOURCE=50
  SEED_URLS=https://eic.ec.europa.eu/eic-funding-opportunities/eic-accelerator_en,https://www.eurekanetwork.org/open-calls/smart-cluster-2025/
"""

import os, json, logging, hashlib, re
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
load_dotenv()

from mistralai import Mistral
import gradio as gr
import requests
from bs4 import BeautifulSoup
from readability import Document

# ---------------------- Setup ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
if not MISTRAL_API_KEY:
    logging.warning("MISTRAL_API_KEY not set. Put it in .env or docker compose environment.")
mclient = Mistral(api_key=MISTRAL_API_KEY)

PRIMARY_ENV = (os.getenv("MODEL_NAME") or "mistral-small-latest").strip()
# Fallback order: your primary -> small -> large -> 8b
MODEL_FALLBACKS: List[str] = []
_seen = set()
for m in [PRIMARY_ENV, "mistral-small-latest", "mistral-large-latest", "ministral-8b-latest"]:
    if m and m not in _seen:
        _seen.add(m)
        MODEL_FALLBACKS.append(m)

TEMPERATURE = 0.2
MAX_TOKENS  = 900
CACHE_PATH = os.getenv("CACHE_PATH", "./calls_cache.json")
SCRAPE_SOURCES = [s.strip() for s in os.getenv("SCRAPE_SOURCES", "seed").split(",") if s.strip()]  # default to seed
MAX_CALLS_PER_SOURCE = int(os.getenv("MAX_CALLS_PER_SOURCE", "50"))

# ---------------------- Tiny Mistral chat helpers ----------------------
def _chat_complete(model: str, messages: List[Dict], **kwargs) -> Optional[str]:
    """
    kwargs: temperature, max_tokens, response_format={"type":"json_object"} for JSON mode
    """
    try:
        resp = mclient.chat.complete(model=model, messages=messages, **kwargs)
        msg = resp.choices[0].message
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            return "".join([p.get("text", "") for p in content if isinstance(p, dict)])
        return str(content) if content is not None else None
    except Exception as e:
        logging.warning(f"[mistral.chat] {model} -> {e}")
        return None

def mistral_text(prompt: str, temperature: float = 0.2, max_tokens: int = 900) -> Optional[str]:
    for model in MODEL_FALLBACKS:
        txt = _chat_complete(
            model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        if txt:
            return txt
    return None

def mistral_json(prompt: str, max_tokens: int = 1024) -> Optional[dict]:
    """
    Ask for strict JSON; if the SDK or model returns text around it, extract the first JSON object.
    """
    for model in MODEL_FALLBACKS:
        txt = _chat_complete(
            model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        if not txt:
            continue
        # Try direct JSON parse
        try:
            return json.loads(txt)
        except Exception:
            pass
        # Fallback: extract first {...}
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
    return None

# ---------------------- Minimal data model ----------------------
@dataclass
class CallRecord:
    id: str
    source: str
    title: str
    url: str
    deadline_date: Optional[str]
    open_date: Optional[str]
    awarding_body: Optional[str]
    funding_rate: Optional[float]
    award_min: Optional[float]
    award_max: Optional[float]
    regions: List[str]
    sectors: List[str]
    trl_min: Optional[int]
    trl_max: Optional[int]
    requires_consortium: Optional[bool]
    min_partners: Optional[int]
    ai_relevance: float
    must_haves: List[str]
    summary: str
    raw_excerpt: str
    last_seen: str

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

# ---------------------- Scraping ----------------------
HEADERS = {
  "User-Agent": "GrantMatchLite/1.0 (+edu use)",
  "Accept": "text/html,application/xhtml+xml",
  "Accept-Language": "en-US,en;q=0.9,sv;q=0.8",
}

def _fetch(url: str, timeout: int = 20) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code >= 400:
            logging.warning(f"Fetch {url} -> {r.status_code}")
            return None
        return r.text
    except Exception as e:
        logging.warning(f"Fetch error {url}: {e}")
        return None

def _readable_text(html: str) -> str:
    try:
        doc = Document(html)
        body_html = doc.summary()
    except Exception:
        body_html = html
    soup = BeautifulSoup(body_html, "html.parser")
    txt = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", txt)[:20000]

def _hash_id(source: str, url: str) -> str:
    return hashlib.sha1(f"{source}|{url}".encode("utf-8")).hexdigest()[:16]

def scrape_vinnova(max_items: int = 50) -> List[Tuple[str, str]]:
    pages = ["https://www.vinnova.se/en/calls/", "https://www.vinnova.se/utlysningar/"]
    items, seen = [], set()
    for listing_url in pages:
        html = _fetch(listing_url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            title = a.get_text(strip=True)
            if not title or len(title) < 5:
                continue
            if "/calls/" in href or "/utlysningar/" in href:
                if href.startswith("/"):
                    href = "https://www.vinnova.se" + href
                key = (title, href)
                if key in seen:
                    continue
                seen.add(key)
                items.append((title, href))
                if len(items) >= max_items:
                    return items
    return items

def scrape_eu_fund(max_items: int = 50) -> List[Tuple[str, str]]:
    # JS-heavy portal; leave empty for demo reliability (use seeds/addurl)
    return []

def seed_urls_from_env() -> List[Tuple[str, str]]:
    raw = os.getenv("SEED_URLS", "").strip()
    if not raw:
        return []
    out = []
    for u in [x.strip() for x in raw.split(",") if x.strip()]:
        html = _fetch(u)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else (soup.find("h1").get_text(strip=True) if soup.find("h1") else u)
        out.append((title, u))
    return out

# ---------------------- Normalize with Mistral ----------------------
NORM_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "eligibility_summary": {"type": "array", "items": {"type": "string"}},
        "key_constraints": {"type": "array", "items": {"type": "string"}},
        "budget": {
            "type": "object",
            "properties": {
                "funding_rate": {"type": "number"},
                "award_min": {"type": "number"},
                "award_max": {"type": "number"}
            }
        },
        "deadlines": {
            "type": "object",
            "properties": {
                "open_date": {"type": "string"},
                "deadline_date": {"type": "string"}
            }
        },
        "requires_consortium": {"type": "boolean"},
        "min_partners": {"type": "integer"},
        "trl_min": {"type": "integer"},
        "trl_max": {"type": "integer"},
        "sectors": {"type": "array", "items": {"type": "string"}},
        "ai_relevance": {"type": "number"},
        "must_have_phrases": {"type": "array", "items": {"type": "string"}},
        "awarding_body": {"type": "string"},
        "raw_excerpt": {"type": "string"}
    },
    "required": ["title"]
}

def _fallback_minimal_record(source, url, title_hint, text) -> CallRecord:
    return CallRecord(
        id=_hash_id(source, url),
        source=source,
        title=title_hint or "Untitled",
        url=url,
        deadline_date=None, open_date=None,
        awarding_body=None, funding_rate=None,
        award_min=None, award_max=None,
        regions=["EU"] if source in ("eufund", "seed") else ["SE"],
        sectors=[], trl_min=None, trl_max=None,
        requires_consortium=None, min_partners=None,
        ai_relevance=0.0, must_haves=[],
        summary=(text[:240] + "…") if text else "",
        raw_excerpt=(text[:380] + "…") if text else "",
        last_seen=_now_iso(),
    )

def normalize_call(source: str, url: str, title_hint: str) -> Optional[CallRecord]:
    html = _fetch(url)
    if not html:
        return None
    text = _readable_text(html)

    prompt = f"""
You are a grants analyst for students and early-stage startups in Sweden/EU.
Extract structured data from this funding call page text. If a field is not explicit, return null.
Return ONLY JSON with the schema provided.

TEXT (truncated to ~20k chars):
\"\"\"{text[:20000]}\"\"\" 

Source: {source}
URL: {url}

Schema (informal guidance):
- sectors: short tags like ["AI","Energy","Health","Mobility","GovTech","Education","Climate"]
- ai_relevance: 1=central to AI/data, 0.5=optional track, 0=unrelated
- funding_rate is a fraction (e.g., 0.7 for 70%)
- dates in ISO if present (YYYY-MM-DD), else null
- raw_excerpt: 2-3 sentences directly quoted from the page (<=400 chars)
Return JSON only, no extra text.
"""
    j = mistral_json(prompt, max_tokens=1024)
    if not j:
        logging.warning("LLM normalization failed — using fallback minimal record.")
        return _fallback_minimal_record(source, url, title_hint, text)

    def _get(d, k, default=None):
        v = d.get(k, default) if isinstance(d, dict) else default
        return v

    budget = _get(j, "budget", {}) or {}
    deadlines = _get(j, "deadlines", {}) or {}

    rec = CallRecord(
        id=_hash_id(source, url),
        source=source,
        title=j.get("title") or title_hint or "Untitled",
        url=url,
        deadline_date=_get(deadlines, "deadline_date"),
        open_date=_get(deadlines, "open_date"),
        awarding_body=j.get("awarding_body"),
        funding_rate=_get(budget, "funding_rate"),
        award_min=_get(budget, "award_min"),
        award_max=_get(budget, "award_max"),
        regions=["EU"] if source in ("eufund", "seed") else ["SE"],
        sectors=j.get("sectors") or [],
        trl_min=j.get("trl_min"),
        trl_max=j.get("trl_max"),
        requires_consortium=j.get("requires_consortium"),
        min_partners=j.get("min_partners"),
        ai_relevance=float(j.get("ai_relevance") or 0.0),
        must_haves=j.get("must_have_phrases") or [],
        summary=" • ".join((j.get("eligibility_summary") or [])[:4]),
        raw_excerpt=(j.get("raw_excerpt") or "")[:400],
        last_seen=_now_iso(),
    )
    return rec

# ---------------------- Cache ----------------------
def load_cache(path: str = CACHE_PATH) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_cache(cache: Dict[str, Dict], path: str = CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def refresh_sources(sources: List[str]) -> List[CallRecord]:
    logging.info(f"Refreshing sources: {sources}")
    found: List[Tuple[str,str,str]] = []  # (source, title, url)

    if "vinnova" in sources:
        for title, url in scrape_vinnova(MAX_CALLS_PER_SOURCE):
            found.append(("vinnova", title, url))

    if "eufund" in sources:
        for title, url in scrape_eu_fund(MAX_CALLS_PER_SOURCE):
            found.append(("eufund", title, url))

    # Always include seeds so demo works even if listings fail
    for title, url in seed_urls_from_env():
        found.append(("seed", title, url))

    logging.info(f"Discovered {len(found)} candidate pages. Normalizing...")

    recs: List[CallRecord] = []
    for src, title, url in found:
        try:
            rec = normalize_call(src, url, title)
            if rec:
                recs.append(rec)
        except Exception as e:
            logging.warning(f"Normalize failed {url}: {e}")

    by_id: Dict[str, CallRecord] = {}
    for r in recs:
        by_id[r.id] = r
    logging.info(f"Normalized {len(by_id)} calls.")

    cache = {rid: asdict(rec) for rid, rec in by_id.items()}
    save_cache(cache, CACHE_PATH)
    return list(by_id.values())

# ---------------------- Search/Filter ----------------------
def load_records() -> List[CallRecord]:
    cache = load_cache()
    recs: List[CallRecord] = []
    for rid, d in cache.items():
        try:
            recs.append(CallRecord(**d))
        except Exception:
            pass
    return recs

def filter_records(recs: List[CallRecord], query: str, ai_only: bool, max_results: int = 30) -> List[CallRecord]:
    q = (query or "").lower().strip()
    scored: List[Tuple[float, CallRecord]] = []
    for r in recs:
        hay = " ".join([r.title, r.summary or "", " ".join(r.sectors), r.raw_excerpt or ""]).lower()
        base = 0.0
        if not q:
            base = 0.1
        else:
            for tok in set(q.split()):
                if tok and tok in hay:
                    base += 1.0
        if ai_only and (r.ai_relevance or 0.0) < 0.5:
            continue
        scored.append((base + (r.ai_relevance or 0.0)*0.2, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in scored[:max_results]]

# ---------------------- Chat logic ----------------------
SYSTEM_PROMPT = (
    "You are a grants concierge for students and early-stage startups in Sweden and the EU. "
    "Answer ONLY using the provided dataset of scraped calls. "
    "When unsure or if a detail isn't in the dataset, say you don't know and suggest visiting the official link. "
    "Always include short, friendly, and precise answers; end with 1–2 bullet action tips when relevant. "
    "Cite sources inline as [n] with the call title; provide its URL after the answer under 'Sources'."
)

def _build_context(recs: List[CallRecord]) -> Tuple[str, List[Tuple[str, str]]]:
    lines, citations = [], []
    for i, r in enumerate(recs, start=1):
        citations.append((r.title.strip(), r.url))
        line = [
            f"[{i}] {r.title.strip()} — source={r.source}",
            f"URL: {r.url}",
            f"Deadline: {r.deadline_date or 'unknown'}; Open: {r.open_date or 'unknown'}",
            f"Funding rate: {r.funding_rate if r.funding_rate is not None else 'unknown'}; Award: {r.award_min or '?'}–{r.award_max or '?'}",
            f"Sectors: {', '.join(r.sectors) or 'unknown'}; TRL: {r.trl_min or '?'}–{r.trl_max or '?'}",
            f"Consortium required: {r.requires_consortium}; Min partners: {r.min_partners}",
            f"Must-haves: {', '.join(r.must_haves[:5])}",
            f"Summary: {r.summary}",
            "---"
        ]
        lines.append("\n".join(line))
    return "\n".join(lines), citations

GENERAL_PROMPT = (
    "You are a helpful grants adviser for EU/Sweden. "
    "If the user asks about specific calls but no dataset is available, provide general guidance only: "
    "key programmes (e.g., Horizon Europe, EIC Accelerator, Eurostars/Eureka, Vinnova), typical co-funding rates, TRL fit, "
    "eligibility patterns (SME-only, consortiums), deadlines cadence, and application tips. "
    "Do NOT invent specific call details or dates. Prefer short, structured answers with 3–5 action bullets."
)

def general_fund_answer(user_q: str) -> str:
    txt = mistral_text(
        f"{GENERAL_PROMPT}\n\nUser question: {user_q}",
        temperature=0.2,
        max_tokens=600
    )
    if txt:
        return txt
    return ("I can share general guidance on EU/SE funding (programmes like Horizon Europe, "
            "EIC Accelerator, Eurostars, and Vinnova), selection criteria, and application tips. Ask me a question!")

def answer_question(user_q: str, k: int = 8, ai_only: bool = False) -> str:
    recs = load_records()
    if not recs:
        # Fallback conversation when dataset is empty
        msg = "**No scraped dataset yet — answering with general funding guidance.**\n\n"
        return msg + general_fund_answer(user_q)

    top = filter_records(recs, user_q, ai_only=ai_only, max_results=k)
    if not top:
        return "I couldn't find a relevant call in the current dataset. Try different keywords or type **refresh**."

    ctx, cites = _build_context(top)
    prompt = f"""{SYSTEM_PROMPT}

DATASET:
{ctx}

Question: {user_q}

Answer requirements:
- Be concise (<= 8 sentences unless listing options).
- If listing matches, sort by earliest deadline first.
- Cite the most relevant items by [index] in-line.
- After the answer, print a 'Sources' section with '• [index] Title — URL' on separate lines.
"""
    txt = mistral_text(prompt, temperature=0.2, max_tokens=900)
    if txt:
        src_lines = ["\n**Sources**:"]
        for i, (t, u) in enumerate(cites, start=1):
            src_lines.append(f"• [{i}] {t} — {u}")
        return txt.strip() + "\n" + "\n".join(src_lines)
    else:
        logging.error("Mistral text generation failed.")
        return "Sorry, I couldn't process that question."

# ---------------------- Gradio UI ----------------------
INTRO = "👋 I'm your grant concierge. **What are your next funding steps?**\n\nType `refresh` to load sources."

def _route_message(user_text: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
    reply = None
    t = (user_text or "").strip()

    if t.lower() in {"refresh", "/refresh"}:
        recs = refresh_sources(SCRAPE_SOURCES)
        reply = f"Refreshed {len(recs)} calls from: {', '.join(SCRAPE_SOURCES)}.\nAsk a question!"
    elif t.lower().startswith("ai only"):
        q = t[len("ai only"):].strip() or "AI funding"
        reply = answer_question(q, ai_only=True)
    elif t.lower().startswith("addurl "):
        url = t.split(" ", 1)[1].strip()
        if not url.startswith("http"):
            reply = "Please provide a full URL (https://...)."
        else:
            html = _fetch(url)
            if not html:
                reply = "Couldn't fetch that URL."
            else:
                soup = BeautifulSoup(html, "html.parser")
                title = soup.title.get_text(strip=True) if soup.title else (soup.find("h1").get_text(strip=True) if soup.find("h1") else url)
                rec = normalize_call("manual", url, title)
                if not rec:
                    reply = "I couldn't extract structured info from that page."
                else:
                    cache = load_cache()
                    cache[rec.id] = asdict(rec)
                    save_cache(cache, CACHE_PATH)
                    reply = f"Added: {rec.title}\nNow ask a question!"
    else:
        reply = answer_question(t, ai_only=False)

    history = history + [(user_text, reply)]
    return history, ""  # clear textbox

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# 🎓💶 GrantMatch Lite — Student & Startup Funding Concierge")

    # Remove unsupported kwargs like 'autofocus'
    chatbot = gr.Chatbot(height=520)
    msg = gr.Textbox(placeholder="Type your question (try: refresh, ai only ... , addurl <link>)")
    clear = gr.Button("Clear")

    # Initial assistant greeting to verify readiness
    def _init_chat():
        # Chatbot expects list of (user, assistant) tuples; empty user, assistant message
        return [("", INTRO)]
    demo.load(_init_chat, inputs=None, outputs=chatbot)

    def on_submit(message, history):
        return _route_message(message, history)

    msg.submit(on_submit, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: [("", INTRO)], None, chatbot)

if __name__ == "__main__":
    share_flag = str(os.getenv("GRADIO_SHARE", "false")).lower() == "true"
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share_flag)
