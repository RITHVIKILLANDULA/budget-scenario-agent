"""Rule-based question -> scenario draft.

There is no model call in here. Every mapping from English to a field is a
pattern plus the alias lists in vocab.py, which means the whole language layer
is inspectable and testable, and it runs with no API key. It also means it is
brittle in the ways rule systems are brittle: the trace it returns is there so
you can see exactly which words it used and which it ignored.

The parser only produces a *draft* dict. Validation is a separate step, so a
sentence can parse cleanly and still be rejected.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from . import fiscal, vocab

MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}

DECREASE_VERBS = ("cut", "reduce", "trim", "lower", "drop", "slash", "shrink", "decrease", "save", "pull back", "scale back", "take out")
INCREASE_VERBS = ("increase", "raise", "grow", "add", "boost", "expand", "bump", "invest", "scale up", "ramp")
FREEZE_VERBS = ("freeze", "hold flat", "hold", "flat-line", "flatline", "cap", "keep flat", "no growth")
SHIFT_VERBS = ("shift", "move", "reallocate", "redirect", "insource", "swap")

FRACTIONS = {
    "half": 50.0, "a half": 50.0, "a third": 100 / 3, "one third": 100 / 3,
    "two thirds": 200 / 3, "a quarter": 25.0, "one quarter": 25.0,
    "three quarters": 75.0, "a tenth": 10.0,
}

STOPWORDS = set(
    """what whats happens happen if we our the a an to of in on for by over from with
    and then also plus but while when do does can could should would will show me tell
    about is are it its their there this that next year years budget spend spending
    costs cost much how many at as per each just only still even so please run model
    scenario case be been being have has had get got make making across during into
    out up down more less than total across-the-board ok okay us""".split()
)


@dataclass
class ParseResult:
    draft: dict | None
    errors: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    ignored_words: list[str] = field(default_factory=list)
    tokens_total: int = 0
    tokens_matched: int = 0

    @property
    def coverage(self) -> float:
        return self.tokens_matched / self.tokens_total if self.tokens_total else 0.0

    @property
    def ok(self) -> bool:
        return self.draft is not None and not self.errors


# --- alias index -------------------------------------------------------------

def _alias_index() -> list[tuple[str, str, str]]:
    """(alias, kind, canonical key), longest alias first so 'marketing programs'
    wins over 'marketing'."""
    entries: list[tuple[str, str, str]] = []
    for key, meta in vocab.CATEGORIES.items():
        entries.append((vocab.category_label(key).lower(), "category", key))
        entries.append((key.replace("_", " "), "category", key))
        for alias in meta["aliases"]:
            entries.append((alias, "category", key))
    for key, meta in vocab.DEPARTMENTS.items():
        entries.append((key.lower(), "department", key))
        for alias in meta["aliases"]:
            entries.append((alias, "department", key))
    for spec in vocab.VENDORS:
        entries.append((spec.name.lower(), "vendor", spec.name))
        for alias in spec.aliases:
            entries.append((alias, "vendor", spec.name))
    seen: set[tuple[str, str, str]] = set()
    unique = [e for e in entries if not (e in seen or seen.add(e))]
    return sorted(unique, key=lambda e: -len(e[0]))


ALIASES = _alias_index()

# A word that is both a department and a category name ("marketing") is read as
# the department only when it follows one of these.
DEPT_PREPOSITIONS = ("in", "for", "across", "within", "inside", "at")


def _find_targets(clause: str, trace: list[str]) -> tuple[dict, set[int]]:
    consumed: set[int] = set()
    found = {"departments": [], "categories": [], "vendors": []}
    for alias, kind, key in ALIASES:
        for match in re.finditer(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", clause):
            span = set(range(match.start(), match.end()))
            if span & consumed:
                continue
            resolved_kind = kind
            if kind in ("department", "category"):
                other = "category" if kind == "department" else "department"
                collides = any(a == alias and k == other for a, k, _ in ALIASES)
                if collides:
                    before = clause[: match.start()].strip().split()
                    prep = before[-1] if before else ""
                    resolved_kind = "department" if prep in DEPT_PREPOSITIONS else "category"
                    trace.append(
                        f"'{alias}' matches both a department and a category; read as "
                        f"{resolved_kind} because the preceding word is "
                        f"{prep!r} (use 'in {alias}' to force the department)"
                    )
                    if resolved_kind != kind:
                        continue  # the other entry in the index will pick it up
            bucket = {"department": "departments", "category": "categories", "vendor": "vendors"}[resolved_kind]
            target_key = key
            if target_key not in found[bucket]:
                found[bucket].append(target_key)
                trace.append(f"matched {resolved_kind} {target_key!r} from {alias!r}")
            consumed |= span
    return found, consumed


# --- magnitudes --------------------------------------------------------------

_MONEY = re.compile(
    r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*([kmb])?|\b(\d[\d,]{2,}(?:\.\d+)?)\s*(dollars|usd)\b",
    re.I,
)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct\b|points?\b)")


def _money(clause: str) -> tuple[float | None, tuple[int, int] | None]:
    m = _MONEY.search(clause)
    if not m:
        return None, None
    raw = (m.group(1) or m.group(3)).replace(",", "")
    value = float(raw)
    suffix = (m.group(2) or "").lower()
    value *= {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(suffix, 1)
    return value, m.span()


def _percent(clause: str) -> tuple[float | None, tuple[int, int] | None]:
    m = _PERCENT.search(clause)
    if m:
        return float(m.group(1)), m.span()
    for phrase, value in sorted(FRACTIONS.items(), key=lambda kv: -len(kv[0])):
        m = re.search(rf"\bby {re.escape(phrase)}\b|\b{re.escape(phrase)} of\b", clause)
        if m:
            return value, m.span()
    return None, None


# --- window ------------------------------------------------------------------

def _window(clause: str, trace: list[str]) -> tuple[dict, set[int]]:
    consumed: set[int] = set()

    m = re.search(r"\bq([1-4])\b|\bquarter ([1-4])\b", clause)
    if m:
        q = int(m.group(1) or m.group(2))
        consumed |= set(range(*m.span()))
        trace.append(
            f"period Q{q} resolved on the July-June fiscal calendar to "
            f"{fiscal.quarter_label(q)}"
        )
        return {"kind": "quarter", "quarter": q}, consumed

    m = re.search(r"\b(first|second|third|fourth|1st|2nd|3rd|4th) quarter\b", clause)
    if m:
        q = ORDINALS[m.group(1)]
        consumed |= set(range(*m.span()))
        trace.append(f"period resolved to {fiscal.quarter_label(q)}")
        return {"kind": "quarter", "quarter": q}, consumed

    m = re.search(r"\bh([12])\b|\b(first|second|front|back) half\b", clause)
    if m:
        half = int(m.group(1)) if m.group(1) else (1 if m.group(2) in ("first", "front") else 2)
        consumed |= set(range(*m.span()))
        months = fiscal.half_months(half)
        trace.append(
            f"period H{half} resolved to {fiscal.label(months[0])}..{fiscal.label(months[-1])}"
        )
        return {"kind": "half", "half": half}, consumed

    months: list[dt.date] = []
    for name, number in MONTH_NAMES.items():
        for m in re.finditer(rf"\b{name}\b", clause):
            if len(name) <= 3 and not re.search(rf"\b{name}\b(?!\w)", clause):
                continue
            plan_month = next(
                (d for d in fiscal.PLAN_MONTHS if d.month == number), None
            )
            if plan_month and plan_month not in months:
                months.append(plan_month)
                consumed |= set(range(*m.span()))
    if months:
        months.sort()
        trace.append(
            "period read as named months "
            + ", ".join(fiscal.label(d) for d in months)
        )
        return {"kind": "months", "months": months}, consumed

    trace.append(
        f"no period in the question; applied to the whole plan year "
        f"FY{fiscal.PLAN_FY} ({fiscal.label(fiscal.PLAN_MONTHS[0])}.."
        f"{fiscal.label(fiscal.PLAN_MONTHS[-1])})"
    )
    return {"kind": "year"}, consumed


# --- clause splitting --------------------------------------------------------

_ACTIONISH = re.compile(
    r"\b(" + "|".join(re.escape(v.split()[0]) for v in
                      DECREASE_VERBS + INCREASE_VERBS + FREEZE_VERBS + SHIFT_VERBS) + r")\w*\b"
)


def _split_clauses(text: str) -> list[str]:
    parts = re.split(r";|\bwhile\b|\bat the same time\b|,? and then\b", text)
    out: list[str] = []
    for part in parts:
        pieces = re.split(r"\band\b", part)
        buffer = pieces[0]
        for piece in pieces[1:]:
            if _ACTIONISH.search(piece):
                out.append(buffer)
                buffer = piece
            else:
                buffer += " and " + piece
        out.append(buffer)
    return [p.strip() for p in out if p.strip()]


def _kind_and_sign(clause: str, trace: list[str]) -> tuple[str, int, set[int]]:
    consumed: set[int] = set()
    for verb in SHIFT_VERBS:
        m = re.search(rf"\b{verb}\w*\b", clause)
        if m and re.search(r"\b(to|into)\b", clause[m.end():]):
            consumed |= set(range(*m.span()))
            trace.append(f"verb {m.group(0)!r} -> shift lever")
            return "shift", 1, consumed
    for verb in FREEZE_VERBS:
        m = re.search(rf"\b{re.escape(verb)}\w*\b", clause)
        if m:
            consumed |= set(range(*m.span()))
            trace.append(f"verb {m.group(0)!r} -> freeze lever")
            return "freeze", 0, consumed
    for verb in DECREASE_VERBS:
        m = re.search(rf"\b{re.escape(verb)}\w*\b", clause)
        if m:
            consumed |= set(range(*m.span()))
            trace.append(f"verb {m.group(0)!r} -> reduction")
            return "change", -1, consumed
    for verb in INCREASE_VERBS:
        m = re.search(rf"\b{re.escape(verb)}\w*\b", clause)
        if m:
            consumed |= set(range(*m.span()))
            trace.append(f"verb {m.group(0)!r} -> increase")
            return "change", 1, consumed
    return "change", 0, consumed


def _shift_destination(clause: str, trace: list[str]) -> tuple[str | None, set[int]]:
    m = re.search(r"\b(?:to|into)\b\s+(.{2,40})$", clause)
    tail = m.group(1) if m else ""
    for alias, kind, key in ALIASES:
        if kind != "category":
            continue
        hit = re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", tail)
        if hit:
            trace.append(f"shift destination {key!r} from {alias!r}")
            start = m.start(1) + hit.start()
            return key, set(range(start, start + len(alias)))
    return None, set()


def _scenario_name(draft: dict) -> str:
    lever = draft["levers"][0]
    target = lever["target"]
    scope = (
        (target["vendors"] or target["categories"] or target["departments"] or ["all spend"])[0]
    )
    scope = vocab.CATEGORIES[scope]["label"] if scope in vocab.CATEGORIES else scope
    window = lever["window"]
    when = {
        "year": f"FY{fiscal.PLAN_FY}",
        "quarter": f"Q{window.get('quarter')}",
        "half": f"H{window.get('half')}",
        "months": "selected months",
    }[window["kind"]]
    if lever["kind"] == "shift":
        destination = vocab.CATEGORIES[lever["to_category"]]["label"]
        return f"{scope} -> {destination} {lever['pct']:.4g}% {when}"[:80]
    if lever["kind"] == "freeze":
        magnitude = "freeze"
    elif lever["kind"] == "absolute_monthly":
        magnitude = f"{lever['amount_usd_per_month']/1000:+,.0f}k/mo"
    else:
        magnitude = f"{lever['pct']:+.4g}%"
    extra = f" +{len(draft['levers']) - 1} more" if len(draft["levers"]) > 1 else ""
    return f"{scope} {magnitude} {when}{extra}"[:80]


def parse_question(question: str) -> ParseResult:
    text = question.lower().strip()
    if not text:
        return ParseResult(None, errors=["empty question"])

    trace: list[str] = []
    errors: list[str] = []
    levers: list[dict] = []
    consumed_words: set[str] = set()

    clauses = _split_clauses(text)
    if len(clauses) > 1:
        trace.append(f"split into {len(clauses)} clauses: " + " | ".join(clauses))

    for clause in clauses:
        clause_trace: list[str] = []
        kind, sign, consumed = _kind_and_sign(clause, clause_trace)
        targets, target_spans = _find_targets(clause, clause_trace)
        consumed |= target_spans
        window, window_spans = _window(clause, clause_trace)
        consumed |= window_spans

        lever: dict = {"target": targets, "window": window}

        if kind == "freeze":
            lever["kind"] = "freeze"
        elif kind == "shift":
            destination, dest_span = _shift_destination(clause, clause_trace)
            consumed |= dest_span
            pct, pct_span = _percent(clause)
            if pct is None:
                errors.append(
                    f"'{clause.strip()}' asks to shift spend but does not say how much"
                )
                continue
            consumed |= set(range(*pct_span))
            # the destination category must not also be read as the source
            if destination and destination in targets["categories"]:
                targets["categories"] = [c for c in targets["categories"] if c != destination]
            lever.update(kind="shift", pct=pct, to_category=destination)
        else:
            pct, pct_span = _percent(clause)
            money, money_span = _money(clause)
            if sign == 0 and (pct is not None or money is not None):
                sign = -1
                clause_trace.append("no verb found; a bare magnitude is read as a cut")
            if pct is not None:
                consumed |= set(range(*pct_span))
                lever.update(kind="percent_change", pct=sign * pct)
            elif money is not None:
                consumed |= set(range(*money_span))
                monthly_match = re.search(r"\b(per month|a month|monthly|/month|/mo|each month)\b", clause)
                monthly = bool(monthly_match)
                if monthly_match:
                    consumed |= set(range(*monthly_match.span()))
                months = len(_resolve_window_months(window))
                amount = money if monthly else money / months
                if not monthly:
                    clause_trace.append(
                        f"${money:,.0f} has no period, so it is read as a total over "
                        f"the {months}-month window = ${amount:,.0f}/month"
                    )
                lever.update(kind="absolute_monthly", amount_usd_per_month=sign * amount)
            else:
                errors.append(
                    f"'{clause.strip()}' has no size -- give a percentage "
                    f"(\"15%\"), a dollar amount (\"$40k a month\"), or say \"freeze\""
                )
                trace.extend(clause_trace)
                continue

        if lever["kind"] != "freeze" and not any(targets.values()):
            if re.search(r"\b(everything|all spend|opex|across the board|total spend)\b", clause):
                clause_trace.append("no named target; 'across the board' applies to every line")
            else:
                errors.append(
                    f"'{clause.strip()}' does not name a department, category or "
                    f"vendor I hold. Known categories: "
                    f"{', '.join(vocab.category_label(c) for c in vocab.CATEGORY_KEYS)}."
                )
                trace.extend(clause_trace)
                continue

        levers.append(lever)
        trace.extend(clause_trace)
        for index in sorted(consumed):
            consumed_words.add(clause[index])
        # record the literal consumed substrings for the coverage count
        lever["_consumed"] = "".join(clause[i] if i in consumed else " " for i in range(len(clause)))

    tokens = [t for t in re.findall(r"[a-z0-9$%&']+", text) if t not in STOPWORDS]
    used = " ".join(l.pop("_consumed", "") for l in levers)
    used_tokens = set(re.findall(r"[a-z0-9$%&']+", used))
    ignored = [t for t in tokens if not any(t in u or u in t for u in used_tokens if u)]

    if not levers:
        return ParseResult(
            None,
            errors=errors or ["nothing in that question mapped to a budget lever"],
            trace=trace,
            ignored_words=ignored,
            tokens_total=len(tokens),
            tokens_matched=len(tokens) - len(ignored),
        )

    draft = {
        "name": "",
        "question": question.strip(),
        "levers": levers,
        "fiscal_year": fiscal.PLAN_FY,
    }
    draft["name"] = _scenario_name(draft)
    if ignored:
        trace.append("words I did not use: " + ", ".join(sorted(set(ignored))))

    return ParseResult(
        draft=draft,
        errors=errors,
        trace=trace,
        ignored_words=ignored,
        tokens_total=len(tokens),
        tokens_matched=len(tokens) - len(ignored),
    )


def _resolve_window_months(window: dict) -> list[dt.date]:
    kind = window["kind"]
    if kind == "quarter":
        return fiscal.quarter_months(window["quarter"])
    if kind == "half":
        return fiscal.half_months(window["half"])
    if kind == "months":
        return list(window["months"])
    return fiscal.PLAN_MONTHS
