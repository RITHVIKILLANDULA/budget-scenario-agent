"""Optional model-backed parsing.

The rules parser in `parser.py` is the one that always works. This module is
the other half of the trade-off that README argues about: when an API key is
present, the question goes to a model first, and the model's job is narrow --
propose the same draft dict the rules parser emits, using only the vocabulary
that is actually in the ledger. Nothing else changes. `validate` still runs,
the schema still decides, and the arithmetic is still Python.

The model therefore cannot produce a number a user sees. It can only produce a
request, and a request that does not survive `Scenario.model_validate` is
dropped on the floor -- the question falls back to the rules parser and the
user is none the wiser.

With no key this module returns None immediately and costs nothing. That is
the default path: a clean clone, and the public deploy, have no key.

Configuration, all optional:

    GROQ_API_KEY    enables the model path
    GROQ_MODEL      default qwen/qwen3.8-27b
    GROQ_BASE_URL   any OpenAI-compatible endpoint
    GROQ_TIMEOUT    seconds, default 12
    BUDGET_LLM=off  ignore the key and stay on rules

Read from the environment or from a `.env` file next to `app.py`, which is
gitignored. No key is ever written to disk by this code.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from . import fiscal, vocab
from .models import Scenario
from .parser import scenario_name

DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_TIMEOUT = 12.0
MAX_LEVERS = 6

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

_LEVER_KINDS = ("percent_change", "absolute_monthly", "freeze", "shift")


# --- configuration -----------------------------------------------------------

def _load_env_file(path: Path = ENV_FILE) -> None:
    """Populate os.environ from a .env file, never overriding what is set."""
    try:
        text = path.read_text()
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT


def config() -> LLMConfig | None:
    """The model configuration, or None when the model path is off."""
    _load_env_file()
    if os.environ.get("BUDGET_LLM", "").strip().lower() in ("off", "0", "false", "no"):
        return None
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not key:
        return None
    try:
        timeout = float(os.environ.get("GROQ_TIMEOUT", "") or DEFAULT_TIMEOUT)
    except ValueError:
        timeout = DEFAULT_TIMEOUT
    return LLMConfig(
        api_key=key,
        model=(os.environ.get("GROQ_MODEL") or "").strip() or DEFAULT_MODEL,
        base_url=(os.environ.get("GROQ_BASE_URL") or "").strip() or DEFAULT_BASE_URL,
        timeout=timeout,
    )


def status() -> dict:
    """What the sidebar needs to know. Never contains the key."""
    cfg = config()
    if cfg is None:
        return {"enabled": False, "model": None, "detail": "GROQ_API_KEY is not set"}
    return {"enabled": True, "model": cfg.model, "detail": f"{cfg.model} via {cfg.base_url}"}


# --- the prompt --------------------------------------------------------------

SYSTEM_PROMPT = """\
You convert a budget question into a structured scenario request for a
planning model that does its own arithmetic. You never compute, estimate or
mention any dollar figure or result. You only describe the change the question
is asking for, using the exact vocabulary you are given.

Reply with one JSON object and nothing else:

{"levers": [ ... ]}

Each lever is one change, applied in the order you list them:

  {"kind": "percent_change", "pct": -15,
   "target": {"departments": [], "categories": ["contractors"], "vendors": []},
   "window": {"kind": "quarter", "quarter": 3}}

  "kind" is one of:
    percent_change     needs "pct", negative to cut, positive to increase,
                       between -100 and 200
    absolute_monthly   needs "amount_usd_per_month", negative to cut; if the
                       question gives a total, divide it by the months in the
                       window yourself
    freeze             takes no magnitude; holds the run rate flat
    shift              needs "pct" (positive) and "to_category"; the target
                       must name the source category or vendor

  "target" selects ledger lines. Empty lists mean every line. Departments,
  categories and vendors intersect. Use the exact keys and names listed below
  and nothing else.

  "window" is one of:
    {"kind": "year"}
    {"kind": "quarter", "quarter": 1-4}       fiscal quarters, see the calendar
    {"kind": "half", "half": 1-2}
    {"kind": "months", "months": ["2026-10-01", "2026-11-01"]}

Rules:
- Use only the exact category keys, department names and vendor names given.
  Never invent, translate or abbreviate one. A synonym in the question should
  be mapped to the vocabulary item it means.
- If the question names something that is not in the vocabulary, or asks for
  something none of the four levers can express (an optimisation, a headcount
  change, a question about the past, a request for advice), return
  {"levers": []}. An empty list is the correct answer to anything you cannot
  express exactly. Never approximate.
- No prose, no explanation, no markdown fence, no trailing commas."""


def vocabulary_block() -> str:
    """The real chart of accounts and the real date range, as text."""
    categories = "\n".join(
        f"  {key:<22} {vocab.category_label(key)}"
        for key in vocab.CATEGORY_KEYS
    )
    vendors = "\n".join(
        f"  {spec.name:<26} {spec.category}" for spec in vocab.VENDORS
    )
    plan = fiscal.PLAN_MONTHS
    quarters = "\n".join(
        f"  Q{q}  {fiscal.label(fiscal.quarter_months(q)[0])}.."
        f"{fiscal.label(fiscal.quarter_months(q)[-1])}"
        for q in (1, 2, 3, 4)
    )
    return f"""\
Category keys (use the key on the left, never the label):
{categories}

Department names (use them exactly as written):
{"".join(f"  {d}" + chr(10) for d in vocab.DEPARTMENT_KEYS)}\
Vendor names (use them exactly as written), and the category each sits in:
{vendors}

Calendar. The fiscal year starts in July, so fiscal Q3 is January to March.
  Plan year        FY{fiscal.PLAN_FY}  {fiscal.label(plan[0])}..{fiscal.label(plan[-1])}
{quarters}
  H1  {fiscal.label(fiscal.half_months(1)[0])}..{fiscal.label(fiscal.half_months(1)[-1])}
  H2  {fiscal.label(fiscal.half_months(2)[0])}..{fiscal.label(fiscal.half_months(2)[-1])}
  Actuals on file  {fiscal.label(fiscal.HISTORY_START)}..{fiscal.label(fiscal.HISTORY_END)}
Every window must fall inside the plan year. There is nothing to plan outside it."""


def build_prompt(question: str) -> str:
    return f"{vocabulary_block()}\n\nQuestion: {question.strip()}"


# --- the call ----------------------------------------------------------------

class LLMError(RuntimeError):
    pass


def _post(cfg: LLMConfig, prompt: str) -> str:
    body = json.dumps(
        {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
        }
    ).encode()
    request = urllib.request.Request(
        cfg.base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
            # the endpoint sits behind a CDN that rejects the stdlib default
            "User-Agent": "budget-scenario-agent",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise LLMError(f"HTTP {exc.code}") from exc
    except Exception as exc:  # timeout, DNS, TLS, malformed body -- all the same here
        raise LLMError(f"{type(exc).__name__}") from exc
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("response had no message content") from exc


@lru_cache(maxsize=256)
def _cached_post(model: str, base_url: str, prompt: str) -> str:
    """Streamlit re-runs the whole script on every widget change, and the
    compare panel asks two more questions on each of those. Same question,
    same prompt, same answer -- so call once."""
    cfg = config()
    if cfg is None:
        raise LLMError("no key")
    return _post(cfg, prompt)


# --- reading the reply -------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?|```\s*$", re.M)


def extract_json(text: str) -> dict:
    """The reply as a dict, fence or no fence, prose around it or not."""
    cleaned = _FENCE.sub("", text or "").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMError("reply was not JSON")
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError("reply was not JSON") from exc
    if not isinstance(value, dict):
        raise LLMError("reply was not a JSON object")
    return value


def _names(value, known: tuple[str, ...], kind: str) -> list[str]:
    """Reject anything the ledger does not hold, before Pydantic sees it."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise LLMError(f"{kind} was not a list")
    lookup = {k.lower(): k for k in known}
    out = []
    for item in value:
        if not isinstance(item, str):
            raise LLMError(f"{kind} entry was not a string")
        resolved = lookup.get(item.strip().lower())
        if resolved is None:
            raise LLMError(f"{item!r} is not a {kind} in the ledger")
        if resolved not in out:
            out.append(resolved)
    return out


def _number(value, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMError(f"{field_name} was not a number")
    return float(value)


def _window(value) -> dict:
    if value is None:
        return {"kind": "year"}
    if not isinstance(value, dict):
        raise LLMError("window was not an object")
    kind = value.get("kind", "year")
    if kind == "quarter":
        return {"kind": "quarter", "quarter": int(_number(value.get("quarter"), "quarter"))}
    if kind == "half":
        return {"kind": "half", "half": int(_number(value.get("half"), "half"))}
    if kind == "months":
        months = value.get("months") or []
        if not isinstance(months, list) or not months:
            raise LLMError("months window had no months")
        # "2026-10" is the shape the calendar is printed in; the schema wants a date
        return {
            "kind": "months",
            "months": [
                f"{m}-01" if isinstance(m, str) and re.fullmatch(r"\d{4}-\d{2}", m) else m
                for m in months
            ],
        }
    if kind == "year":
        return {"kind": "year"}
    raise LLMError(f"unknown window kind {kind!r}")


def normalise(payload: dict) -> list[dict]:
    """The model's JSON as a list of draft levers, or an exception."""
    levers = payload.get("levers")
    if levers is None and payload.get("kind"):  # a bare lever, generously read
        levers = [payload]
    if not isinstance(levers, list):
        raise LLMError("no levers in the reply")
    if not levers:
        raise LLMError("the model declined to express this as levers")
    if len(levers) > MAX_LEVERS:
        raise LLMError(f"{len(levers)} levers is more than this tool runs")

    drafts = []
    for raw in levers:
        if not isinstance(raw, dict):
            raise LLMError("a lever was not an object")
        kind = raw.get("kind")
        if kind not in _LEVER_KINDS:
            raise LLMError(f"unknown lever kind {kind!r}")
        target = raw.get("target") or {}
        if not isinstance(target, dict):
            raise LLMError("target was not an object")
        draft: dict = {
            "kind": kind,
            "target": {
                "departments": _names(target.get("departments"), vocab.DEPARTMENT_KEYS, "department"),
                "categories": _names(target.get("categories"), vocab.CATEGORY_KEYS, "category"),
                "vendors": _names(target.get("vendors"), vocab.VENDOR_KEYS, "vendor"),
            },
            "window": _window(raw.get("window")),
        }
        # Magnitudes are copied across whenever the reply carries them, in the
        # kind they arrived in, and the schema decides whether that kind of
        # lever is allowed to have them. A freeze with a percentage attached
        # is a misunderstanding, not something to tidy up quietly.
        if raw.get("pct") is not None:
            draft["pct"] = _number(raw["pct"], "pct")
        if raw.get("amount_usd_per_month") is not None:
            draft["amount_usd_per_month"] = _number(
                raw["amount_usd_per_month"], "amount_usd_per_month"
            )
        if raw.get("to_category") is not None:
            destination = _names(raw["to_category"], vocab.CATEGORY_KEYS, "category")
            draft["to_category"] = destination[0] if destination else None
        drafts.append(draft)
    return drafts


# --- the public entry point --------------------------------------------------

@dataclass
class LLMParse:
    """What the model path returned, and why, for the trace."""

    draft: dict | None
    model: str
    trace: list[str] = field(default_factory=list)


def parse_question(question: str, client=None, cfg: LLMConfig | None = None) -> LLMParse | None:
    """Ask the model for a draft scenario.

    Returns None when the model path is off, so the caller can skip it without
    knowing anything about this module. Returns an LLMParse with `draft=None`
    when the model answered but the answer did not survive the schema: the
    caller falls back to rules, and the reason lands in the trace rather than
    in front of the user.

    `client` is any callable taking the prompt and returning the raw reply. It
    exists so the tests can exercise this whole path with no network.
    """
    if client is None:
        cfg = cfg or config()
        if cfg is None:
            return None
    model = cfg.model if cfg else "stub"
    prompt = build_prompt(question)

    try:
        reply = client(prompt) if client is not None else _cached_post(
            cfg.model, cfg.base_url, prompt
        )
        levers = normalise(extract_json(reply))
        draft = {
            "name": "",
            "question": question.strip(),
            "levers": levers,
            "fiscal_year": fiscal.PLAN_FY,
        }
        draft["name"] = scenario_name(draft)
        Scenario.model_validate(draft)  # the gate, run here so a failure is silent
    except LLMError as exc:
        return LLMParse(None, model, [f"model parse discarded: {exc}"])
    except ValidationError as exc:
        reasons = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'scenario'}: {err['msg']}"
            for err in exc.errors()[:3]
        )
        return LLMParse(None, model, [f"model parse failed the schema: {reasons}"])
    except Exception as exc:  # a broken reply must never take the app down
        return LLMParse(None, model, [f"model parse discarded: {type(exc).__name__}"])

    return LLMParse(
        draft,
        model,
        [
            f"{model} proposed {len(levers)} lever(s); the schema accepted them",
            *(f"model lever: {lever['kind']} {lever['target']}" for lever in levers),
        ],
    )
