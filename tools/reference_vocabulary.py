#!/usr/bin/env python3
"""Extract the reference numerals a patent's own text names.

A patent description is a closed list of the reference numerals that may
legitimately appear in its drawings: EPO drafting practice names every one in
prose ("the housing 104", "valve 116a"). Stage 0 never consulted it, which is
why 34.5% of the numerals OCR read confidently on the curated cohort appear
nowhere in the patent -- readings like '8, 8, 8, 8' in a drawing whose
numerals run 100, 102, 116.

The vocabulary turns three open problems into one constrained one: OCR becomes
closed-set, a confident misreading becomes detectable, and every accepted
label carries a link to the sentence that describes it.

    python -m tools.reference_vocabulary --patent EP1467415B1
    python -m tools.reference_vocabulary --all --output output/PatentData/vocabularies

Written per patent as <patent>_vocabulary.json:

    {"schema": ..., "patent_id": ..., "series": "100x",
     "tokens": {"104": {"mentions": 39, "nouns": ["housing", "body"],
                        "sources": ["description"]}, ...}}
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

SCHEMA = "ap3-reference-vocabulary-v1"
VERSION = "1"

# A reference numeral: 1-4 digits with an optional leading letter ("A42") and
# an optional trailing letter or prime ("14a", "20A", "14'"). Both suffix cases
# occur in the cohort and a digits-only pattern misses them entirely.
_TOKEN = r"[A-Z]?\d{1,4}(?:[A-Za-z]|'|’|''|”)?"

# "the housing 104", "a valve 116a", "second arm 14'"
_NOUN_NUM = re.compile(rf"\b([a-z][a-z-]{{2,}})\s+({_TOKEN})(?![\d\w])", re.I)
# "(104)", "(104, 106)"
_PAREN = re.compile(rf"\(\s*({_TOKEN}(?:\s*,\s*{_TOKEN})*)\s*\)")
# "104, 106 and 108" - enumerations hanging off a single noun
_ENUM = re.compile(rf"\b({_TOKEN})(?:\s*,\s*({_TOKEN}))+(?:\s*(?:and|or)\s+({_TOKEN}))?", re.I)

# Words that precede a number without making it a reference numeral.
_NOT_A_REFERENCE_NOUN = {
    "claim", "claims", "fig", "figs", "figure", "figures", "sheet", "page",
    "step", "steps", "example", "embodiment", "table", "formula", "equation",
    "item", "section", "paragraph", "line", "column", "col", "no", "nos",
    "number", "numbers", "type", "class", "group", "part",
    # units and measures
    "mm", "cm", "um", "nm", "kg", "mg", "ml", "hz", "khz", "mhz", "ghz",
    "degrees", "degree", "percent", "wt", "vol", "ppm", "psi", "bar", "rpm",
    "about", "approximately", "least", "most", "than", "between", "from",
    "to", "up", "over", "under", "within", "and", "or", "of", "in", "at",
}
# Units that FOLLOW a number.
_UNIT_AFTER = re.compile(r"^\s*(?:mm|cm|m|um|µm|nm|kg|g|mg|ml|l|hz|khz|mhz|ghz|"
                         r"%|°|deg|degrees?|wt|vol|ppm|psi|bar|rpm|s|ms|min|h)\b", re.I)


# Elements whose numbers are never reference numerals: figure callouts, cited
# literature (ISSNs, page ranges), addresses, and tabulated data. EPO files
# nest citations inside the description, so dropping them by element is the
# only reliable way -- "Beekstraat 1" and "ISSN 0013" both reach the text
# otherwise.
_DROP_ELEMENTS = ("figref", "citation", "nplcit", "patcit", "addressbook",
                  "tables", "table", "maths", "math", "chemistry", "img")


def _strip_markup(xml: str) -> str:
    xml = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", xml, flags=re.S | re.I)
    for element in _DROP_ELEMENTS:
        xml = re.sub(rf"<{element}\b[^>]*>.*?</{element}>", " ", xml, flags=re.S | re.I)
        xml = re.sub(rf"<{element}\b[^>]*/>", " ", xml, flags=re.I)
    return re.sub(r"<[^>]+>", " ", xml)


def _sections(xml: str) -> dict[str, str]:
    """Description and claims separately; a numeral seen only in claims is weaker."""
    out = {}
    for name, pattern in (("description", r"<description\b.*?</description>"),
                          ("claims", r"<claims\b.*?</claims>")):
        blocks = re.findall(pattern, xml, flags=re.S | re.I)
        out[name] = _strip_markup(" ".join(blocks)) if blocks else ""
    if not out["description"] and not out["claims"]:
        out["description"] = _strip_markup(xml)      # flat file, no sectioning
    return out


def _normalise(token: str) -> str:
    return token.replace("’", "'").replace("”", "''").strip()


def _plausible(noun: str, token: str, tail: str) -> bool:
    if noun.lower() in _NOT_A_REFERENCE_NOUN:
        return False
    if _UNIT_AFTER.match(tail):
        return False
    digits = re.search(r"\d+", token).group(0)
    if len(digits) == 4 and 1850 <= int(digits) <= 2100:
        return False                                  # a year
    return True


def _collect(text: str, source: str, tokens: dict) -> None:
    for match in _NOUN_NUM.finditer(text):
        noun, token = match.group(1), _normalise(match.group(2))
        if not _plausible(noun, token, text[match.end():match.end() + 12]):
            continue
        entry = tokens.setdefault(token, {"mentions": 0, "nouns": collections.Counter(),
                                          "sources": set()})
        entry["mentions"] += 1
        entry["nouns"][noun.lower()] += 1
        entry["sources"].add(source)
    for match in _PAREN.finditer(text):
        for token in re.findall(_TOKEN, match.group(1)):
            token = _normalise(token)
            entry = tokens.setdefault(token, {"mentions": 0, "nouns": collections.Counter(),
                                              "sources": set()})
            entry["mentions"] += 1
            entry["sources"].add(source)
    # An enumeration inherits plausibility from its first member, which the
    # noun pattern has usually already accepted.
    for match in _ENUM.finditer(text):
        members = [_normalise(t) for t in re.findall(_TOKEN, match.group(0))]
        if not members or members[0] not in tokens:
            continue
        for token in members[1:]:
            entry = tokens.setdefault(token, {"mentions": 0, "nouns": collections.Counter(),
                                              "sources": set()})
            entry["mentions"] += 1
            entry["sources"].add(source)


def _series(tokens: dict) -> tuple[str | None, int | None]:
    """The digit width the patent numbers in, weighted by how often each is used.

    Counting distinct tokens lets a handful of stray one-digit matches outvote
    twenty-seven genuine three-digit numerals. Mentions are the better signal:
    a real reference numeral is named repeatedly.
    """
    weights = collections.Counter()
    for token, entry in tokens.items():
        weights[len(re.search(r"\d+", token).group(0))] += entry["mentions"]
    if not weights:
        return None, None
    width, count = weights.most_common(1)[0]
    if count < 0.6 * sum(weights.values()):
        return None, None                             # genuinely mixed
    return {1: "1x", 2: "10x", 3: "100x", 4: "1000x"}.get(width), width


def build(xml_paths: list[Path], patent_id: str, min_mentions: int = 1) -> dict:
    tokens: dict[str, dict] = {}
    xml = " ".join(p.read_text(errors="ignore") for p in xml_paths)
    for source, text in _sections(xml).items():
        if text:
            _collect(text, source, tokens)
    kept = {}
    for token, entry in tokens.items():
        # A token seen once with no noun attached is a page number as often as
        # a reference; require either repetition or a noun.
        if entry["mentions"] < min_mentions or (entry["mentions"] < 2 and not entry["nouns"]):
            continue
        kept[token] = {
            "mentions": entry["mentions"],
            "nouns": [n for n, _ in entry["nouns"].most_common(4)],
            "sources": sorted(entry["sources"]),
        }
    series, width = _series(kept)
    if width is not None:
        # Off-series tokens survive only if the text leans on them; a 100x
        # patent that mentions "5" twice is describing something else.
        kept = {t: e for t, e in kept.items()
                if len(re.search(r"\d+", t).group(0)) == width or e["mentions"] >= 5}
    return {
        "schema": SCHEMA, "version": VERSION, "patent_id": patent_id,
        "sources": [str(p) for p in xml_paths],
        "series": series,
        "n_tokens": len(kept),
        "tokens": dict(sorted(kept.items(), key=lambda kv: (-kv[1]["mentions"], kv[0]))),
    }


def load(path: Path) -> dict:
    """Read a vocabulary; raise rather than degrade into a permissive empty one."""
    document = json.loads(Path(path).read_text())
    if document.get("schema") != SCHEMA:
        raise ValueError(f"Unexpected vocabulary schema: {document.get('schema')!r}")
    if not document.get("tokens"):
        raise ValueError(f"Vocabulary for {document.get('patent_id')} has no tokens")
    return document


def for_patent(root: Path, patent_id: str) -> dict | None:
    """Build the vocabulary straight from the corpus, or None when there is no XML."""
    paths = sorted((Path(root) / patent_id).glob("*.xml"))
    if not paths:
        return None
    try:
        return build(paths, patent_id)
    except (OSError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path,
                    default=Path("data/PatentData/ReorganisedData"))
    ap.add_argument("--patent", help="one patent id; omit with --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--output", type=Path, help="directory for <patent>_vocabulary.json")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if args.patent:
        patents = [args.patent]
    elif args.all:
        patents = sorted(d.name for d in args.corpus.iterdir() if d.is_dir())
        if args.limit:
            patents = patents[:args.limit]
    else:
        ap.error("pass --patent or --all")

    written = skipped = 0
    sizes = []
    for patent in patents:
        document = for_patent(args.corpus, patent)
        if document is None or not document["tokens"]:
            skipped += 1
            continue
        sizes.append(document["n_tokens"])
        written += 1
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / f"{patent}_vocabulary.json").write_text(
                json.dumps(document, indent=2) + "\n")
        elif len(patents) == 1:
            print(json.dumps(document, indent=2))
    if len(patents) > 1 or args.output:
        import statistics
        print(f"vocabularies written: {written}  skipped (no xml / no tokens): {skipped}")
        if sizes:
            print(f"tokens per patent: median {statistics.median(sizes):.0f} "
                  f"min {min(sizes)} max {max(sizes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
