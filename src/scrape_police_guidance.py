"""Scrape England & Wales police training/guidance.

Sources (Crown copyright / Open Government Licence):
  - College of Policing Authorised Professional Practice (APP): https://www.college.police.uk/app
  - PACE Codes of Practice collection on GOV.UK.

Live HTML pages are cached to data/corpus/police_guidance/. A checked-in curated
reference of key PACE Code provisions (pace_codes_reference.json) sits alongside
them so generation always has reliable Code A/C/G/H anchors even if a live page
layout changes; build_corpus.py reads both.
"""
from __future__ import annotations

import json

from common import GUIDANCE_DIR, fetch, write_text

# (slug, human title, url). Pages are best-effort; failures are skipped.
# NOTE: College of Policing APP deep pages sit behind bot protection (HTTP 403),
# so the substantive policing guidance comes from the curated PACE Codes reference
# (pace_codes_reference.json) plus the GOV.UK PACE collection page below. The APP
# landing page is captured for provenance/attribution only.
PAGES = [
    ("app_landing", "College of Policing: Authorised Professional Practice (landing)",
     "https://www.college.police.uk/app"),
    ("govuk_pace_codes", "GOV.UK: PACE codes of practice",
     "https://www.gov.uk/guidance/police-and-criminal-evidence-act-1984-pace-codes-of-practice"),
]

LICENCE = "Open Government Licence v3.0 (Crown copyright)"


def main() -> None:
    manifest = []
    for slug, title, url in PAGES:
        print(f"[guidance] {title} <- {url}")
        try:
            resp = fetch(url, accept="text/html")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! failed: {exc}")
            continue
        out = GUIDANCE_DIR / f"{slug}.html"
        write_text(out, resp.text)
        manifest.append(
            {"slug": slug, "title": title, "url": url, "file": out.name, "licence": LICENCE}
        )
        print(f"  saved {out.name} ({len(resp.text):,} bytes)")

    write_text(GUIDANCE_DIR / "manifest.json", json.dumps(manifest, indent=2))
    print(f"[guidance] done: {len(manifest)} live pages "
          f"(+ curated pace_codes_reference.json)")


if __name__ == "__main__":
    main()
