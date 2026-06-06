"""Scrape England & Wales case law from The National Archives 'Find Case Law'.

Source: https://caselaw.nationalarchives.gov.uk  (Open Justice Licence).
Method: Atom search feed to discover judgments, then fetch each judgment's
        LegalDocML XML by appending `/data.xml` to its URI.
Docs:   https://nationalarchives.github.io/ds-find-caselaw-docs/public
"""
from __future__ import annotations

import json
import re
import urllib.parse

from lxml import etree

from common import CASELAW_DIR, fetch, write_text

ATOM = "{http://www.w3.org/2005/Atom}"
BASE = "https://caselaw.nationalarchives.gov.uk"

# Topic searches covering the broad incident mix. Each yields recent judgments.
QUERIES = [
    "stop and search reasonable grounds",
    "section 60 stop and search",
    "arrest necessity section 24 PACE",
    "use of force police proportionate",
    "breach of the peace police",
    "police detention right to legal advice",
    "drink driving breath specimen",
    "police public order section 14",
]

# Known landmark police-powers judgments to try directly (skipped if not present).
# These are the reliable, on-topic core; discovery adds breadth and is keyword
# filtered at corpus-build time so off-topic results are dropped.
SEED_URIS = [
    "uksc/2015/79",      # R (Roberts) v MPC - s.60 stop and search
    "uksc/2017/9",       # R (Hicks) v MPC - preventive detention / breach of peace
    "uksc/2015/49",      # Beghal v DPP - stop powers
    "ukhl/2006/12",      # R (Gillan) v MPC - stop and search safeguards
    "ukhl/2009/5",       # Austin v MPC - detention / kettling / Art 5
    "ewca/civ/2011/911", # Hayes v CC Merseyside - s.24 arrest necessity test
    "ewca/civ/2013/69",  # ZH v MPC - use of force, disability, Art 3/5/8
    "ewca/civ/2009/414", # Wood v MPC - retention of data / Art 8
]

MAX_JUDGMENTS = 25
LICENCE = "Open Justice Licence, The National Archives (Find Case Law)"


def discover_uris() -> list[str]:
    """Return judgment URIs (e.g. 'uksc/2015/79') discovered via Atom search."""
    found: list[str] = []
    seen: set[str] = set()
    for q in QUERIES:
        url = f"{BASE}/atom.xml?query={urllib.parse.quote(q)}&per_page=6"
        print(f"[caselaw] search: {q!r}")
        try:
            resp = fetch(url, accept="application/atom+xml")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! search failed: {exc}")
            continue
        root = etree.fromstring(resp.content)
        for entry in root.iterfind(f"{ATOM}entry"):
            link = entry.find(f"{ATOM}link")
            href = link.get("href") if link is not None else None
            if not href:
                continue
            # Normalise to a path like 'uksc/2015/79'
            path = urllib.parse.urlparse(href).path.strip("/")
            path = re.sub(r"/data\.\w+$", "", path)
            if path and path not in seen:
                seen.add(path)
                found.append(path)
    return found


def fetch_judgment(path: str) -> dict | None:
    url = f"{BASE}/{path}/data.xml"
    try:
        resp = fetch(url, accept="application/akn+xml, application/xml")
    except Exception as exc:  # noqa: BLE001
        print(f"  !! {path}: {exc}")
        return None
    slug = path.replace("/", "_")
    out = CASELAW_DIR / f"{slug}.xml"
    write_text(out, resp.text)
    print(f"  saved {out.name} ({len(resp.text):,} bytes)")
    return {"path": path, "url": url, "file": out.name, "licence": LICENCE}


def main() -> None:
    uris = SEED_URIS + [u for u in discover_uris() if u not in SEED_URIS]
    manifest = []
    for path in uris:
        if len(manifest) >= MAX_JUDGMENTS:
            break
        rec = fetch_judgment(path)
        if rec:
            manifest.append(rec)

    write_text(CASELAW_DIR / "manifest.json", json.dumps(manifest, indent=2))
    print(f"[caselaw] done: {len(manifest)} judgments")


if __name__ == "__main__":
    main()
