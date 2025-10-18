# grant_chatbot.py
"""
GrantMatch Lite - a Mistral + scraping chatbot for student/startup funding calls.

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

import os, json, logging, hashlib, math, re
from typing import Any, List, Dict, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import pandas as pd
import io

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

# ---------------------- User profile memory ----------------------
@dataclass
class StartupProfile:
    name: Optional[str] = None
    idea: Optional[str] = None
    sector_tags: List[str] = None
    digital: Optional[bool] = None
    country: Optional[str] = None
    stage: Optional[str] = None
    capital_need_eur: Optional[float] = None
    timeline_months: Optional[int] = None
    team_size: Optional[int] = None
    business_model: Optional[str] = None
    trl: Optional[int] = None
    keywords: List[str] = None
    saved_recommendations: List[Dict[str, Any]] = None
    pending_reset: bool = False

# ---------------------- Rule-based exclusions ----------------------
PROFILE_PATH = os.getenv("PROFILE_PATH", "/app/data/profile.json")
PROFILE_PERSIST = str(os.getenv("PROFILE_PERSIST", "false")).lower() == "true"
if not PROFILE_PERSIST:
    try:
        if os.path.exists(PROFILE_PATH):
            os.remove(PROFILE_PATH)
            logging.info("Cleared cached startup profile on launch (set PROFILE_PERSIST=true to keep it).")
    except Exception as e:
        logging.warning(f"Could not reset profile cache on startup: {e}")

PROFILE_FIELD_QUESTIONS: List[Tuple[str, str]] = [
    ("name", "What's the name of your startup or project?"),
    ("idea", "Give me your one-liner pitch - what problem are you solving and how?"),
    ("sector_tags", "Which industries do you fit best (e.g., health, cleantech, AI, food)?"),
    ("country", "Where is your startup based or where will you apply from?"),
    ("stage", "Where are you on the journey right now (idea, MVP, pre-seed, seed)?"),
    ("business_model", "How do you plan to make money - what's the business model?"),
    ("team_size", "How big is the core team today?"),
    ("digital", "Is the solution primarily digital or tech-enabled?"),
    ("capital_need_eur", "How much capital are you aiming to secure in euros?"),
    ("timeline_months", "What timeline are you targeting to put the funding to work (in months)?"),
    ("trl", "Do you know your technology readiness level (TRL)? A number between 1-9 is perfect."),
    ("keywords", "Any keywords or themes reviewers should associate with you?"),
]

PROFILE_COMPLETION_FIELDS = [field for field, _ in PROFILE_FIELD_QUESTIONS]
PROFILE_FIELD_LABELS = {
    "name": "the startup name",
    "idea": "the concept",
    "sector_tags": "your sector focus",
    "country": "your location",
    "stage": "your stage",
    "business_model": "the business model",
    "team_size": "team size",
    "digital": "your digital focus",
    "capital_need_eur": "the capital target",
    "timeline_months": "timeline",
    "trl": "TRL",
    "keywords": "keywords",
}


def _value_present(val) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        return val.strip() != ""
    if isinstance(val, list):
        return len([v for v in val if _value_present(v)]) > 0
    return True


def _render_profile_value(field: str, value) -> str:
    if field == "sector_tags" and isinstance(value, list):
        return ", ".join(value)
    if field == "capital_need_eur" and isinstance(value, (int, float)):
        return f"about EUR {int(value):,}"
    if field == "digital" and isinstance(value, bool):
        return "mainly digital" if value else "more physical than digital"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def profile_missing_fields(p: StartupProfile) -> List[Tuple[str, str]]:
    missing: List[Tuple[str, str]] = []
    for field, question in PROFILE_FIELD_QUESTIONS:
        if not _value_present(getattr(p, field, None)):
            missing.append((field, question))
    return missing


def profile_ready(p: StartupProfile) -> bool:
    return all(_value_present(getattr(p, field, None)) for field in PROFILE_COMPLETION_FIELDS)


def profile_completion_ratio(p: StartupProfile) -> float:
    total = len(PROFILE_COMPLETION_FIELDS)
    if total == 0:
        return 1.0
    completed = sum(1 for field in PROFILE_COMPLETION_FIELDS if _value_present(getattr(p, field, None)))
    return completed / total


def build_profile_question(p: StartupProfile, changed_fields: List[str]) -> Optional[str]:
    missing = profile_missing_fields(p)
    if not missing:
        return None
    _, question = missing[0]
    acknowledgement = ""
    for field in changed_fields:
        value = getattr(p, field, None)
        if _value_present(value):
            label = PROFILE_FIELD_LABELS.get(field, field.replace("_", " "))
            acknowledgement = f"Great, noted {label}: {_render_profile_value(field, value)}. "
            break
    if not acknowledgement and _value_present(p.idea):
        acknowledgement = "Got it - that vision sounds exciting. "
    return f"{acknowledgement}{question}"

def print_startup_profile(profile: StartupProfile):
    try:
        logging.info("Startup profile snapshot: %s", profile_summary(profile))
        print("=== Current Startup Profile ===")
        print(json.dumps(asdict(profile), indent=2, ensure_ascii=False))
        print("================================")
        print(flush=True)
    except Exception as e:
        logging.warning(f"Could not print startup profile: {e}")

def _empty_profile() -> StartupProfile:
    return StartupProfile(sector_tags=[], keywords=[], saved_recommendations=[], pending_reset=False)

def load_profile() -> StartupProfile:
    try:
        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            return StartupProfile(**{**_empty_profile().__dict__, **d})
    except Exception as e:
        logging.warning(f"Load profile failed: {e}")
    return _empty_profile()

def save_profile(p: StartupProfile):
    try:
        os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in p.__dict__.items()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Save profile failed: {e}")

def _merge_profile(old: StartupProfile, new: StartupProfile) -> StartupProfile:
    merged = StartupProfile(**old.__dict__)
    for k, v in new.__dict__.items():
        if v in (None, "", []):
            continue
        if isinstance(v, list):
            if k in {"sector_tags", "keywords"}:
                existing = list(merged.__dict__.get(k) or [])
                existing_lower = {str(item).strip().lower() for item in existing if isinstance(item, str)}
                for item in v:
                    item_str = str(item).strip()
                    if item_str and item_str.lower() not in existing_lower:
                        existing.append(item_str)
                        existing_lower.add(item_str.lower())
                merged.__dict__[k] = existing
            else:
                merged.__dict__[k] = v
        else:
            merged.__dict__[k] = v
    return merged

def profile_summary(p: StartupProfile) -> str:
    parts = []
    if p.name: parts.append(f"Name: {p.name}")
    if p.idea: parts.append(f"Idea: {p.idea}")
    if p.sector_tags: parts.append(f"Sectors: {', '.join(p.sector_tags)}")
    if p.country: parts.append(f"Country: {p.country}")
    if p.stage: parts.append(f"Stage: {p.stage}")
    if p.business_model: parts.append(f"Model: {p.business_model}")
    if p.team_size: parts.append(f"Team: {p.team_size}")
    if p.digital is not None: parts.append(f"Digital: {p.digital}")
    if p.capital_need_eur: parts.append(f"Capital need: ~EUR {int(p.capital_need_eur):,}")
    if p.timeline_months: parts.append(f"Timeline: {p.timeline_months} months")
    if p.trl: parts.append(f"TRL: {p.trl}")
    if p.keywords: parts.append(f"Keywords: {', '.join(p.keywords[:8])}")
    return " | ".join(parts) if parts else "No profile captured yet."

PROFILE_SCHEMA_HINT = """
Return a compact JSON object with any fields you can infer (omit unknowns):
{
  "name": string?,
  "idea": string?,
  "sector_tags": string[]?,   // e.g., ["Food","Retail","Franchise","AI","Health"]
  "digital": boolean?,        // true if primarily digital/tech product
  "country": string?,         // ISO-like or free text (e.g., "SE" or "Sweden")
  "stage": string?,           // idea, MVP, pre-seed, seed, Series A
  "capital_need_eur": number?,// approximate if mentioned
  "timeline_months": number?,
  "team_size": number?,
  "business_model": string?,  // B2C, B2B, etc.
  "trl": number?,             // 1..9 if meaningful
  "keywords": string[]?
}
"""

def extract_profile_from_text(text: str) -> Optional[StartupProfile]:
    if not text or len(text.strip()) < 3:
        return None
    prompt = f"""
You update a startup applicant profile. Read the user's message and extract structured fields.

USER MESSAGE:
\"\"\"{text.strip()}\"\"\"

{PROFILE_SCHEMA_HINT}
Return JSON only.
"""
    j = mistral_json(prompt, max_tokens=400)
    if not j:
        return None
    # Normalize lists
    sect = j.get("sector_tags") or []
    keys = j.get("keywords") or []
    try:
        return StartupProfile(
            name=j.get("name"),
            idea=j.get("idea"),
            sector_tags=[s for s in sect if isinstance(s, str)],
            digital=j.get("digital"),
            country=j.get("country"),
            stage=j.get("stage"),
            capital_need_eur=float(j["capital_need_eur"]) if j.get("capital_need_eur") is not None else None,
            timeline_months=int(j["timeline_months"]) if j.get("timeline_months") is not None else None,
            team_size=int(j["team_size"]) if j.get("team_size") is not None else None,
            business_model=j.get("business_model"),
            trl=int(j["trl"]) if j.get("trl") is not None else None,
            keywords=[s for s in keys if isinstance(s, str)],
        )
    except Exception:
        # Be forgiving if types are off
        return StartupProfile(
            name=j.get("name"),
            idea=j.get("idea"),
            sector_tags=[str(s) for s in sect],
            digital=bool(j.get("digital")) if j.get("digital") is not None else None,
            country=j.get("country"),
            stage=j.get("stage"),
            capital_need_eur=None,
            timeline_months=None,
            team_size=None,
            business_model=j.get("business_model"),
            trl=None,
            keywords=[str(s) for s in keys],
        )

def rule_based_exclusion(rec: CallRecord, p: StartupProfile) -> Optional[str]:
    title = (rec.title or "").lower()
    sectors = [s.lower() for s in (rec.sectors or [])]

    # Example: EIT Digital requires digital innovation focus
    if "eit digital" in title or ("eit" in title and "digital" in title):
        if p.digital is False:     # user is explicitly non-digital
            return "EIT Digital focuses on digital innovation; your venture is non-digital."
        # If user didn't specify digital, but sectors clearly food/retail, we can still warn:
        if p.digital is None and any(s in sectors for s in ["food","retail","hospitality","franchise"]) and not any("digital" in s for s in sectors):
            return "EIT Digital is specialized in digital innovation; your described focus seems non-digital."

    # Add more quick rules here as needed, e.g. deeptech-only calls vs. low-tech food ventures
    return None

# ---------------------- CSV Loader ----------------------

def load_from_csv(path: str = "funding_single_column.csv") -> List[CallRecord]:
    if not os.path.exists(path):
        logging.warning(f"CSV not found: {path}")
        return []

    raw_data = open(path, 'r', encoding="utf-8").readlines()
    processed_data = [raw_data[0]]
    for line in raw_data[1:]:
        processed_line = line.strip()[1:-1].replace('""', '"')
        processed_data.append(processed_line + '\n')
    processed_string = "".join(processed_data)

    df = pd.read_csv(io.StringIO(processed_string), sep=',', quotechar='"')

    recs: List[CallRecord] = []
    for _, row in df.iterrows():
        recs.append(
            CallRecord(
                id=_hash_id("csv", row.get("url", str(_))),
                source="csv",
                title=row.get("title", "Untitled"),
                url=row.get("url", ""),
                deadline_date=row.get("deadline_date"),
                open_date=row.get("open_date"),
                awarding_body=row.get("awarding_body"),
                funding_rate=row.get("funding_rate"),
                award_min=row.get("award_min"),
                award_max=row.get("award_max"),
                regions=[row.get("region")] if row.get("region") else [],
                sectors=[row.get("sector")] if row.get("sector") else [],
                trl_min=row.get("trl_min"),
                trl_max=row.get("trl_max"),
                requires_consortium=row.get("requires_consortium"),
                min_partners=row.get("min_partners"),
                ai_relevance=float(row.get("ai_relevance") or 0.0),
                must_haves=[row.get("must_have")] if row.get("must_have") else [],
                summary=row.get("summary", ""),
                raw_excerpt=row.get("raw_excerpt", ""),
                last_seen=_now_iso(),
            )
        )
    return recs


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
        logging.warning("LLM normalization failed - using fallback minimal record.")
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

    # 0) Start with any structured records from CSV / Sheet (no LLM normalize needed)
    recs: List[CallRecord] = []
    csv_path = os.getenv("CSV_PATH", "").strip()
    if csv_path:
        try:
            csv_recs = load_from_csv(csv_path)
            recs.extend(csv_recs)
            logging.info(f"Loaded {len(csv_recs)} records from CSV: {csv_path}")
        except Exception as e:
            logging.warning(f"CSV import failed ({csv_path}): {e}")

    sheet_url = os.getenv("SHEET_CSV_URL", "").strip()
    if sheet_url:
        try:
            sheet_recs = import_sheet_csv(sheet_url)  # if you added the helper; else remove this block
            recs.extend(sheet_recs)
            logging.info(f"Loaded {len(sheet_recs)} records from Google Sheet CSV")
        except Exception as e:
            logging.warning(f"Sheet CSV import failed: {e}")

    # Track existing IDs/URLs to avoid redundant normalizations
    existing_ids = {r.id for r in recs}
    existing_urls = {r.url for r in recs}

    # 1) Discover candidate web pages (scrapers + seeds)
    found: List[Tuple[str, str, str]] = []  # (source, title, url)

    if "vinnova" in sources:
        try:
            for title, url in scrape_vinnova(MAX_CALLS_PER_SOURCE):
                if url not in existing_urls:
                    found.append(("vinnova", title, url))
        except Exception as e:
            logging.warning(f"Vinnova discovery failed: {e}")

    if "eufund" in sources:
        try:
            for title, url in scrape_eu_fund(MAX_CALLS_PER_SOURCE):
                if url not in existing_urls:
                    found.append(("eufund", title, url))
        except Exception as e:
            logging.warning(f"EU discovery failed: {e}")

    # Always include seeds so demo works even if listings fail
    try:
        for title, url in seed_urls_from_env():
            if url not in existing_urls:
                found.append(("seed", title, url))
    except Exception as e:
        logging.warning(f"Seed discovery failed: {e}")

    logging.info(f"Discovered {len(found)} candidate pages. Normalizing...")

    # 2) Normalize found web pages with the LLM (fallback minimal record on failure)
    for src, title, url in found:
        try:
            rec = normalize_call(src, url, title)
            if rec:
                if rec.id in existing_ids:
                    # Prefer the newer record (replace)
                    recs = [r for r in recs if r.id != rec.id]
                recs.append(rec)
                existing_ids.add(rec.id)
                existing_urls.add(rec.url)
        except Exception as e:
            logging.warning(f"Normalize failed {url}: {e}")

    # 3) Deduplicate by id (prefer latest in 'recs' order)
    by_id: Dict[str, CallRecord] = {}
    for r in recs:
        by_id[r.id] = r

    logging.info(f"Normalized {len(by_id)} calls.")

    # 4) Persist cache
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

def filter_records(recs: List[CallRecord], query: str, ai_only: bool, max_results: int = 30, profile: Optional[StartupProfile] = None) -> List[Tuple[float, CallRecord]]:
    q = (query or "").lower().strip()
    prof = profile or _empty_profile()
    prof_sectors = {s.lower() for s in (prof.sector_tags or [])}
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

        # Profile-aware boosts
        if prof_sectors:
            overlap = prof_sectors.intersection({s.lower() for s in (r.sectors or [])})
            if overlap:
                base += 1.0  # reward sector overlap
        if prof.digital is False:
            # downrank strongly tech/digital-only calls
            if any("digital" in (s or "").lower() for s in (r.sectors or [])) or "eit digital" in (r.title or "").lower():
                base -= 1.0

        if ai_only and (r.ai_relevance or 0.0) < 0.5:
            continue

        # Tiny bonus for nearer deadlines (if you want)
        if r.deadline_date:
            try:
                # earlier = higher score
                y,m,d = [int(x) for x in r.deadline_date.split("-")]
                urgency = max(0, 10000 - (y*372 + m*31 + d))
                base += min(1.0, urgency/10000.0)
            except Exception:
                pass

        scored.append((base + (r.ai_relevance or 0.0)*0.2, r))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:max_results]

def _score_to_probability(score: float) -> float:
    clamped = max(-8.0, min(8.0, score))
    return 1.0 / (1.0 + math.exp(-clamped))

def _call_to_saved_dict(rec: CallRecord, score: float) -> Dict[str, Any]:
    return {
        "title": rec.title,
        "url": rec.url,
        "source": rec.source,
        "summary": rec.summary,
        "deadline": rec.deadline_date,
        "open_date": rec.open_date,
        "award_min": rec.award_min,
        "award_max": rec.award_max,
        "funding_rate": rec.funding_rate,
        "regions": rec.regions,
        "sectors": rec.sectors,
        "trl_min": rec.trl_min,
        "trl_max": rec.trl_max,
        "requires_consortium": rec.requires_consortium,
        "min_partners": rec.min_partners,
        "must_haves": rec.must_haves,
        "probability": _score_to_probability(score),
    }


def store_recommendations(profile: StartupProfile, scored: List[Tuple[float, CallRecord]]) -> None:
    saved = [_call_to_saved_dict(rec, score) for score, rec in scored[:3]]
    profile.saved_recommendations = saved
    profile.pending_reset = False
    save_profile(profile)
    print_startup_profile(profile)
    logging.info("Stored %d recommendation(s) on profile.", len(saved))


def describe_saved_call(call: Dict[str, Any]) -> str:
    lines = []
    title = call.get("title") or "Unknown opportunity"
    source = call.get("source") or "unknown source"
    lines.append(f"Here's what I have on {title} ({source}):")
    summary = call.get("summary")
    if summary:
        lines.append(summary)

    meta_parts = []
    if call.get("deadline"):
        meta_parts.append(f"Deadline: {call['deadline']}")
    if call.get("open_date"):
        meta_parts.append(f"Opens: {call['open_date']}")
    if call.get("funding_rate") is not None:
        meta_parts.append(f"Funding rate: {call['funding_rate']}%")
    award_min = call.get("award_min")
    award_max = call.get("award_max")
    if award_min or award_max:
        meta_parts.append(f"Award: {award_min or '?'} - {award_max or '?'} EUR")
    if meta_parts:
        lines.append("; ".join(meta_parts))

    sector_list = call.get("sectors") or []
    if sector_list:
        lines.append(f"Sectors: {', '.join(sector_list)}")
    regions = call.get("regions") or []
    if regions:
        lines.append(f"Regions: {', '.join(regions)}")

    trl_min = call.get("trl_min")
    trl_max = call.get("trl_max")
    if trl_min or trl_max:
        lines.append(f"TRL fit: {trl_min or '?'} - {trl_max or '?'}")

    if call.get("requires_consortium"):
        partners = call.get("min_partners")
        partner_txt = f"{partners}+ partners" if partners else "multi-partner consortium"
        lines.append(f"Consortium requirement: {partner_txt}")

    musts = call.get("must_haves") or []
    if musts:
        lines.append(f"Must-haves: {', '.join(musts[:5])}")

    probability = call.get("probability")
    if probability is not None:
        lines.append(f"Match confidence: ~{int(probability * 100)}% based on your profile fit.")

    url = call.get("url")
    if url:
        lines.append(f"Official link: {url}")
    return "\n".join(lines)


def format_other_opportunities(recs: List[Dict[str, Any]]) -> str:
    if not recs:
        return "I don't have any saved opportunities yet - try asking me to refresh the dataset."
    if len(recs) == 1:
        top = recs[0]
        return ("Right now the standout option is "
                f"{top.get('title') or 'this programme'}. Ask for more detail if you'd like to dive deeper.")
    lines = ["Here are the other opportunities I saved for you:"]
    for call in recs[1:]:
        title = call.get("title") or "Untitled programme"
        deadline = call.get("deadline") or "deadline TBC"
        probability = call.get("probability")
        confidence = f" (~{int(probability * 100)}% fit)" if probability is not None else ""
        summary = call.get("summary")
        lines.append(f"- {title} - {deadline}{confidence}")
        if summary:
            lines.append(f"  Summary: {summary}")
        lines.append(f"  Link: {call.get('url')}")
    return "\n".join(lines)


ORDINAL_KEYWORDS = {
    "first": 0,
    "1st": 0,
    "one": 0,
    "primary": 0,
    "second": 1,
    "2nd": 1,
    "next": 1,
    "two": 1,
    "third": 2,
    "3rd": 2,
    "three": 2,
}


def handle_saved_recommendation_followup(user_text: str, profile: StartupProfile) -> Optional[str]:
    recs = profile.saved_recommendations or []
    if not recs:
        return None
    lowered = (user_text or "").lower()

    if any(keyword in lowered for keyword in OTHER_OPPORTUNITY_KEYWORDS):
        return format_other_opportunities(recs)

    target_index: Optional[int] = None
    for keyword, idx in ORDINAL_KEYWORDS.items():
        if keyword in lowered and idx < len(recs):
            target_index = idx
            break
    if target_index is None:
        for idx, call in enumerate(recs):
            title = (call.get("title") or "").lower()
            if title and title in lowered:
                target_index = idx
                break
    if target_index is None:
        if any(keyword in lowered for keyword in DETAIL_KEYWORDS):
            target_index = 0
        else:
            for idx in range(1, min(len(recs), 3) + 1):
                token = str(idx)
                if (
                    f"option {token}" in lowered
                    or f"choice {token}" in lowered
                    or f"recommendation {token}" in lowered
                    or f"rec {token}" in lowered
                    or f"number {token}" in lowered
                ):
                    target_index = idx - 1
                    break
    if target_index is None:
        generic_detail_phrases = ["the fund", "the vc", "that fund", "that opportunity", "the opportunity"]
        if any(phrase in lowered for phrase in generic_detail_phrases):
            target_index = 0
    if target_index is not None and 0 <= target_index < len(recs):
        return describe_saved_call(recs[target_index])
    return None


def _matches_token(text: str, tokens: set) -> bool:
    normalized = (text or "").strip().lower()
    if normalized in tokens:
        return True
    sanitized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    words = sanitized.split()
    if not words:
        return False
    return words[0] in tokens


def handle_reset_confirmation(profile: StartupProfile, user_text: str) -> Optional[Tuple[StartupProfile, str]]:
    if not profile.pending_reset:
        return None
    lowered = (user_text or "").strip().lower()
    if _matches_token(lowered, YES_TOKENS):
        fresh = _empty_profile()
        save_profile(fresh)
        print_startup_profile(fresh)
        return fresh, "Fresh start it is! Tell me about this new idea and we'll rebuild the profile together."
    if _matches_token(lowered, NO_TOKENS):
        profile.pending_reset = False
        save_profile(profile)
        return profile, "No problem - I'll keep your current profile intact. What would you like to discuss next?"
    return profile, "Just let me know with a quick **yes** or **no** if you want to replace your current profile."


# ---------------------- Chat logic ----------------------
SYSTEM_PROMPT = (
    "You are an energetic grants concierge and startup coach for students and early-stage startups in Sweden and the EU. "
    "Use only the dataset provided in the context. "
    "Recommend the single best-fitting call for the profile, explain why it matches, and share two upbeat, concise next steps. "
    "If nothing fits, be transparent and suggest the most useful profile detail or action to gather next."
)


def _build_context(recs: List[CallRecord]) -> Tuple[str, List[Tuple[str, str]]]:
    lines: List[str] = []
    citations: List[Tuple[str, str]] = []
    for i, r in enumerate(recs, start=1):
        citations.append((r.title.strip(), r.url))
        sectors = ", ".join(r.sectors) if r.sectors else "unknown"
        awards = f"{r.award_min or '?'} - {r.award_max or '?'}"
        trl = f"{r.trl_min or '?'} - {r.trl_max or '?'}"
        musts = ", ".join(r.must_haves[:5]) if r.must_haves else "none stated"
        line = [
            f"[{i}] {r.title.strip()} | source={r.source}",
            f"URL: {r.url}",
            f"Deadline: {r.deadline_date or 'unknown'} | Open: {r.open_date or 'unknown'}",
            f"Funding rate: {r.funding_rate if r.funding_rate is not None else 'unknown'} | Award: {awards}",
            f"Sectors: {sectors} | TRL: {trl}",
            f"Consortium required: {r.requires_consortium} | Min partners: {r.min_partners}",
            f"Must-haves: {musts}",
            f"Summary: {r.summary}",
            "---",
        ]
        lines.append("\n".join(line))
    return "\n".join(lines), citations


GENERAL_PROMPT = (
    "You are a helpful grants adviser for the EU and Sweden. "
    "When no dataset is available, provide general guidance: flagship programmes (Horizon Europe, EIC Accelerator, Eurostars/Eureka, Vinnova), "
    "typical co-funding rates, TRL expectations, eligibility themes, deadline cadence, and two application tips. "
    "Keep it short, structured, and factual. Do not fabricate specific call details or dates."
)


def general_fund_answer(user_q: str) -> str:
    txt = mistral_text(
        f"{GENERAL_PROMPT}\n\nUser question: {user_q}",
        temperature=0.2,
        max_tokens=600,
    )
    if txt:
        return txt
    return (
        "I can share general guidance on EU and Swedish funding programmes (Horizon Europe, EIC Accelerator, Eurostars, Vinnova), "
        "what they typically fund, and how to prepare. Ask me anything specific!"
    )


STARTUP_COACH_PROMPT = (
    "You are an upbeat startup coach and fundraising buddy. "
    "Offer clear, concise guidance on the user's question, drawing on lean startup, fundraising, and go-to-market best practices. "
    "Acknowledge useful profile clues when relevant, keep the tone friendly, and finish with one motivating suggestion."
)


def startup_coach_answer(user_q: str, profile: StartupProfile, followup: Optional[str]) -> str:
    prompt = f"""{STARTUP_COACH_PROMPT}

PROFILE SNAPSHOT:
{profile_summary(profile)}

USER MESSAGE:
{user_q or 'No specific question - offer a proactive next step.'}

Response requirements:
- Give focused advice that fits the stage/persona above.
- Use a conversational tone (2-4 sentences).
- End with an encouraging question or action nudge for the founder.
"""
    if followup:
        prompt += f"\nIf it helps, invite them to answer this question next: {followup}\n"

    txt = mistral_text(prompt, temperature=0.4, max_tokens=400)
    if txt:
        return txt
    fallback = "Happy to brainstorm anything from MVP validation to fundraising prep - what's on your mind?"
    if followup:
        fallback += f" Also, could you share: {followup}"
    return fallback


FUNDING_KEYWORDS = [
    "fund", "grant", "finance", "financing", "money", "capital", "investment",
    "investor", "vc", "venture", "call", "opportunity", "funding", "subsidy",
    "apply", "application", "co-fund", "budget", "raise", "raising",
    "recommend", "recommendation", "option"
]

RESET_KEYWORDS = [
    "new idea",
    "new startup",
    "new project",
    "fresh idea",
    "start over",
    "restart",
    "forget previous",
    "fresh start",
    "reset this",
    "another idea",
    "another startup",
    "another start",
    "another start up",
]

DETAIL_KEYWORDS = [
    "tell me more",
    "more detail",
    "details",
    "learn more",
    "deep dive",
    "explain more",
    "walk me through",
]

OTHER_OPPORTUNITY_KEYWORDS = [
    "other opportunity",
    "other opportunities",
    "other option",
    "other options",
    "something else",
    "anything else",
    "another fund",
    "another vc",
    "another option",
    "what else",
    "more options",
    "the others",
]

READINESS_KEYWORDS = [
    "get ready",
    "prepare",
    "preparation",
    "application tips",
    "application advice",
    "submission tips",
    "pitch deck",
    "due diligence",
    "investor readiness",
    "ready for funding",
    "fundraising readiness",
]

ON_TOPIC_HINTS = [
    "startup",
    "start-up",
    "business",
    "venture",
    "product",
    "prototype",
    "mvp",
    "customer",
    "market",
    "go to market",
    "go-to-market",
    "traction",
    "investor",
    "fund",
    "funding",
    "capital",
    "grant",
    "finance",
    "pitch",
    "team",
    "revenue",
    "budget",
    "runway",
    "fundraise",
    "fundraising",
    "call",
    "programme",
    "program",
    "vc",
    "accelerator",
    "incubator",
]

YES_TOKENS = {"yes", "y", "sure", "please", "do it", "confirm", "go ahead", "absolutely"}
NO_TOKENS = {"no", "n", "not yet", "keep it", "cancel", "hold on", "stay", "nope"}


def wants_funding(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    return any(keyword in lowered for keyword in FUNDING_KEYWORDS)


def wants_readiness_support(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in READINESS_KEYWORDS)

def is_on_topic(text: str) -> bool:
    if not text:
        return True
    if wants_funding(text) or wants_readiness_support(text):
        return True
    lowered = text.lower()
    return any(hint in lowered for hint in ON_TOPIC_HINTS)



def answer_question(user_q: str, k: int = 6, ai_only: bool = False) -> str:
    recs = load_records()
    p = load_profile()

    if not profile_ready(p):
        question = build_profile_question(p, [])
        if not question:
            missing = profile_missing_fields(p)
            question = missing[0][1] if missing else "Could you share a bit more about your startup?"
        progress = int(profile_completion_ratio(p) * 100)
        return f"Let's capture a bit more detail first. {question} (profile {progress}% complete)."

    if not recs:
        msg = "**No scraped/CSV dataset yet - answering with general funding guidance.**\n\n"
        return msg + general_fund_answer(user_q)

    scored = filter_records(recs, user_q, ai_only=ai_only, max_results=k, profile=p)
    if not scored:
        return "I couldn't find a relevant call in the current dataset. Try different keywords or type **refresh**."

    suitable: List[Tuple[float, CallRecord]] = []
    excluded: List[Tuple[CallRecord, str]] = []
    for score, r in scored:
        reason = rule_based_exclusion(r, p)
        if reason:
            excluded.append((r, reason))
        else:
            suitable.append((score, r))

    candidates = suitable or scored
    candidate_records = [rec for _, rec in candidates]
    ctx, cites = _build_context(candidate_records)

    primary_score, primary_call = candidates[0]
    match_probability = _score_to_probability(primary_score)
    logging.info(
        "Match probability for %s: %.1f%%",
        primary_call.title,
        match_probability * 100.0,
    )
    store_recommendations(p, candidates)

    prompt = f"""{SYSTEM_PROMPT}

USER PROFILE:
{profile_summary(p)}

DATASET:
{ctx}

User request: {user_q or 'Match the best funding call to this profile.'}

Answer requirements:
- Begin with 'Recommended call: <title>.' using one of the dataset titles.
- Follow with a concise 2-3 sentence overview of the programme (funding type, ticket size, deadlines, geography) using the dataset details.
- Add a short paragraph explaining why this opportunity aligns with the startup profile, referencing at least two profile attributes.
- Add exactly two bullet points under 'Next steps'.
- Mention a brief caveat if there is any major risk or eligibility concern.
- Do not include a Sources section or numbered citations.
- Treat entry [1] in the dataset as the primary recommendation.
"""

    txt = mistral_text(prompt, temperature=0.2, max_tokens=700)
    if not txt:
        return "Sorry, I couldn't process that question."

    lines = [txt.strip()]

    if cites:
        _, primary_url = cites[0]
        if primary_url:
            lines.append(f"Call link: {primary_url}")

    if excluded:
        lines.append("Filtered out:")
        for r, reason in excluded:
            lines.append(f"- {r.title}: {reason}")

    return "\n".join(line for line in lines if line)


def _newly_filled_fields(old: StartupProfile, new: StartupProfile) -> List[str]:
    changed: List[str] = []
    for field in new.__dict__.keys():
        old_val = getattr(old, field, None)
        new_val = getattr(new, field, None)
        if field not in PROFILE_FIELD_LABELS:
            continue
        if not _value_present(old_val) and _value_present(new_val):
            changed.append(field)
    return changed


def _update_profile_from_user_text(user_text: str) -> Tuple[StartupProfile, List[str]]:
    p = load_profile()
    extracted = extract_profile_from_text(user_text)
    if extracted:
        merged = _merge_profile(p, extracted)
        changed = _newly_filled_fields(p, merged)
        save_profile(merged)
        print_startup_profile(merged)
        return merged, changed
    print_startup_profile(p)
    return p, []

def handle_message(message, history):
    text = (message.get("text") if isinstance(message, dict) else str(message)) or ""
    text = text.strip()

    return answer_question(text, ai_only=False)


# ---------------------- Gradio UI ----------------------
INTRO = "👋 I'm your grant concierge. **What are your next funding steps?**\n\nType `refresh` to load sources."

def _route_message(user_text: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
    reply = None
    t = (user_text or "").strip()
    lower_t = t.lower()
    profile = load_profile()

    if lower_t in {"profile", "/profile"}:
        return (history + [(user_text, "Your profile:\n" + profile_summary(profile))], "")

    if lower_t in {"reset profile", "/reset"}:
        fresh = _empty_profile()
        save_profile(fresh)
        print_startup_profile(fresh)
        return (
            history + [
                (user_text, "Profile cleared. Tell me about your startup so I can tailor funding matches.")
            ],
            "",
        )

    confirmation = handle_reset_confirmation(profile, t)
    if confirmation:
        profile, confirmation_reply = confirmation
        history = history + [(user_text, confirmation_reply)]
        return history, ""

    has_profile_data = any(_value_present(getattr(profile, field, None)) for field in PROFILE_COMPLETION_FIELDS)
    if t and any(keyword in lower_t for keyword in RESET_KEYWORDS) and has_profile_data:
        profile.pending_reset = True
        save_profile(profile)
        current_summary = profile_summary(profile)
        prompt = ("You already have a startup profile saved (" + current_summary +
                  "). Do you want to start a fresh one for this new idea? (yes/no)")
        return (history + [(user_text, prompt)], "")

    followup_reply = handle_saved_recommendation_followup(t, profile)
    if followup_reply:
        history = history + [(user_text, followup_reply)]
        return history, ""

    profile, changed_fields = _update_profile_from_user_text(t)

    if lower_t in {"refresh", "/refresh"}:
        recs = refresh_sources(SCRAPE_SOURCES)
        reply = f"Refreshed {len(recs)} calls from: {', '.join(SCRAPE_SOURCES)}.\nAsk a question!"
    elif lower_t.startswith("ai only"):
        q = t[len("ai only"):].strip() or "AI funding"
        reply = answer_question(q, ai_only=True)
    elif lower_t.startswith("addurl "):
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
        missing_fields = profile_missing_fields(profile)
        followup_question = build_profile_question(profile, changed_fields)
        profile_complete = len(missing_fields) == 0
        profile_complete_now = profile_complete and any(f in PROFILE_COMPLETION_FIELDS for f in changed_fields)
        funding_requested = wants_funding(t)

        if wants_readiness_support(t):
            reply = startup_coach_answer(t, profile, followup_question if not profile_complete else None)
        elif not profile_complete:
            progress = int(profile_completion_ratio(profile) * 100)
            question_text = followup_question or (missing_fields[0][1] if missing_fields else "Could you share a quick detail about your startup?")
            if not changed_fields:
                if t and not is_on_topic(t):
                    prefix = "Let's stay focused on your startup funding journey. "
                elif t:
                    prefix = "Thanks! To keep moving, "
                else:
                    prefix = ""
                reply = f"{prefix}{question_text} (profile {progress}% complete)."
            else:
                reply = f"{question_text} (profile {progress}% complete)."
        elif profile_complete and (profile_complete_now or funding_requested or not t):
            query = t if (funding_requested and t) else "Recommend the best funding opportunity for this startup profile."
            reply = answer_question(query, ai_only=False)
        else:
            if t and not is_on_topic(t):
                reply = ("I'm here to support your startup and funding plans. "
                         "Let me know what you're looking for on that front - funding details, saved opportunities, or readiness tips.")
            else:
                reply = startup_coach_answer(t, profile, followup_question)

    history = history + [(user_text, reply)]
    return history, ""  # clear textbox

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# 🎓💶 GrantMatch Lite - Student & Startup Funding Concierge")

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




