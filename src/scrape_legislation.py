"""Scrape England & Wales statutes from legislation.gov.uk as CLML XML.

Source: legislation.gov.uk (Crown copyright, Open Government Licence v3.0).
Method: append `/data.xml` to any legislation URI to get the CLML XML.
        https://www.legislation.gov.uk/developer
"""
from __future__ import annotations

import json

from common import LEGISLATION_DIR, fetch, write_text

# (slug, human title, legislation.gov.uk path). Path is the act root; we fetch
# the whole-act CLML XML and split into sections later in build_corpus.py.
STATUTES = [
    ("pace_1984", "Police and Criminal Evidence Act 1984", "ukpga/1984/60"),
    ("misuse_drugs_1971", "Misuse of Drugs Act 1971", "ukpga/1971/38"),
    ("cjpoa_1994", "Criminal Justice and Public Order Act 1994", "ukpga/1994/33"),
    ("road_traffic_1988", "Road Traffic Act 1988", "ukpga/1988/52"),
    ("criminal_law_1967", "Criminal Law Act 1967", "ukpga/1967/58"),
    ("public_order_1986", "Public Order Act 1986", "ukpga/1986/64"),
    ("human_rights_1998", "Human Rights Act 1998", "ukpga/1998/42"),
    ("terrorism_2000", "Terrorism Act 2000", "ukpga/2000/11"),
    ("mental_health_1983", "Mental Health Act 1983", "ukpga/1983/20"),
]

LICENCE = "Open Government Licence v3.0 (Crown copyright), via legislation.gov.uk"


def main() -> None:
    manifest = []
    for slug, title, path in STATUTES:
        url = f"https://www.legislation.gov.uk/{path}/data.xml"
        print(f"[legislation] {title} <- {url}")
        try:
            resp = fetch(url, accept="application/xml")
        except Exception as exc:  # noqa: BLE001 - log and continue
            print(f"  !! failed: {exc}")
            continue
        out = LEGISLATION_DIR / f"{slug}.xml"
        write_text(out, resp.text)
        manifest.append(
            {"slug": slug, "title": title, "url": url, "file": out.name, "licence": LICENCE}
        )
        print(f"  saved {out.name} ({len(resp.text):,} bytes)")

    write_text(LEGISLATION_DIR / "manifest.json", json.dumps(manifest, indent=2))
    print(f"[legislation] done: {len(manifest)} statutes")


if __name__ == "__main__":
    main()
