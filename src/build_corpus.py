"""Normalise the scraped corpus into citation-tagged snippets for retrieval.

Output: data/corpus_index.jsonl, one snippet per line:
  {
    "id": str,
    "source_type": "legislation" | "caselaw" | "guidance",
    "citation": str,           # canonical, human citation
    "aliases": [str, ...],     # alternative citation strings for matching
    "title": str,              # section heading / case name / topic
    "text": str,               # the grounding text
    "url": str
  }
Every snippet carries at least one verifiable citation so generated outputs can be
cross-checked against it in validate_dataset.py.
"""
from __future__ import annotations

import json
import re

from lxml import etree

from common import (
    CASELAW_DIR,
    CORPUS_INDEX,
    GUIDANCE_DIR,
    LEGISLATION_DIR,
    write_jsonl,
)

LEG_NS = "{http://www.legislation.gov.uk/namespaces/legislation}"

# slug -> (abbreviation, optional second alias used in citations)
ACT_ABBREV = {
    "pace_1984": ("PACE 1984", "Police and Criminal Evidence Act 1984"),
    "misuse_drugs_1971": ("MDA 1971", "Misuse of Drugs Act 1971"),
    "cjpoa_1994": ("CJPOA 1994", "Criminal Justice and Public Order Act 1994"),
    "road_traffic_1988": ("RTA 1988", "Road Traffic Act 1988"),
    "criminal_law_1967": ("Criminal Law Act 1967", "Criminal Law Act 1967"),
    "public_order_1986": ("POA 1986", "Public Order Act 1986"),
    "human_rights_1998": ("HRA 1998", "Human Rights Act 1998"),
    "terrorism_2000": ("Terrorism Act 2000", "Terrorism Act 2000"),
    "mental_health_1983": ("MHA 1983", "Mental Health Act 1983"),
}

# Only keep statute sections relevant to operational police powers (keeps the
# index focused; full acts are huge). A section is kept if its heading or text
# matches any of these, OR its number is explicitly whitelisted per act.
POWER_TERMS = re.compile(
    r"stop|search|arrest|detain|detention|seiz|entry|premises|caution|"
    r"specimen|breath|drink|driving|licence|constable|force|warrant|"
    r"public order|offensive weapon|reasonable (grounds|suspicion)|"
    r"right to|legal advice|solicitor|inform|harass|alarm|distress",
    re.IGNORECASE,
)

# A judgment is kept only if a party is a police/prosecution body (precise signal
# that it concerns police powers) OR it is one of the curated landmark seeds.
CASE_PARTY = re.compile(
    r"Commissioner of Police|Chief Constable|\bConstable\b|"
    r"Director of Public Prosecutions|\bDPP\b|Crown Prosecution Service|"
    r"Metropolitan Police|\bPolice\b",
    re.IGNORECASE,
)
SEED_STEMS = {
    "uksc_2015_79", "uksc_2017_9", "uksc_2015_49",
    "ewca_civ_2011_911", "ewca_civ_2013_69", "ewca_civ_2009_414",
}
CASE_MAX_SNIPPET_PARAS = 6
MAX_SNIPPET_CHARS = 2200


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def local(el) -> str:
    return etree.QName(el).localname


# --------------------------------------------------------------------------- #
# Legislation (CLML)
# --------------------------------------------------------------------------- #
def parse_legislation() -> list[dict]:
    manifest = json.loads((LEGISLATION_DIR / "manifest.json").read_text())
    by_slug = {m["slug"]: m for m in manifest}
    snippets: list[dict] = []

    for slug, meta in by_slug.items():
        path = LEGISLATION_DIR / meta["file"]
        if not path.exists():
            continue
        abbrev, full_title = ACT_ABBREV.get(slug, (meta["title"], meta["title"]))
        root = etree.parse(str(path)).getroot()

        for grp in root.iter(f"{LEG_NS}P1group"):
            pnum_el = grp.find(f"{LEG_NS}P1/{LEG_NS}Pnumber")
            if pnum_el is None:
                pnum_el = grp.find(f".//{LEG_NS}Pnumber")
            section = norm("".join(pnum_el.itertext())) if pnum_el is not None else ""
            if not section:
                continue
            title_el = grp.find(f"{LEG_NS}Title")
            heading = norm("".join(title_el.itertext())) if title_el is not None else ""
            body = norm(" ".join(grp.itertext()))
            if not body:
                continue
            if not (POWER_TERMS.search(heading) or POWER_TERMS.search(body[:600])):
                continue
            text = body[:MAX_SNIPPET_CHARS]
            citation = f"{full_title}, s.{section}"
            aliases = [
                citation,
                f"s.{section} {abbrev}",
                f"section {section} {abbrev}",
                f"section {section} of the {full_title}",
            ]
            snippets.append(
                {
                    "id": f"leg:{slug}:s{section}",
                    "source_type": "legislation",
                    "citation": citation,
                    "aliases": sorted(set(aliases)),
                    "title": heading or f"{full_title} s.{section}",
                    "text": text,
                    "url": f"{meta['url'].replace('/data.xml','')}/section/{section}",
                }
            )
    return snippets


# --------------------------------------------------------------------------- #
# Case law (Akoma Ntoso / LegalDocML)
# --------------------------------------------------------------------------- #
def _meta_value(root, localname: str) -> str:
    for el in root.iter():
        if local(el) == localname:
            t = norm(el.text or "")
            if t:
                return t
    return ""


def parse_caselaw() -> list[dict]:
    snippets: list[dict] = []
    for path in sorted(CASELAW_DIR.glob("*.xml")):
        root = etree.parse(str(path)).getroot()
        cite = _meta_value(root, "cite")
        # case name from FRBRname value attr
        name = ""
        for el in root.iter():
            if local(el) == "FRBRname" and el.get("value"):
                name = norm(el.get("value"))
                break
        if not cite:
            continue
        if path.stem not in SEED_STEMS and not CASE_PARTY.search(name):
            continue  # not a police-powers judgment

        # Collect body paragraph texts.
        paras: list[str] = []
        body = None
        for el in root.iter():
            if local(el) == "judgmentBody":
                body = el
                break
        scope = body if body is not None else root
        for p in scope.iter():
            if local(p) == "p":
                t = norm("".join(p.itertext()))
                if len(t) > 40:
                    paras.append(t)

        short_name = (name.split(" v ")[0][:40] + " v …") if " v " in name else name[:50]
        # Chunk paragraphs into snippets.
        for i in range(0, len(paras), CASE_MAX_SNIPPET_PARAS):
            chunk = " ".join(paras[i : i + CASE_MAX_SNIPPET_PARAS])[:MAX_SNIPPET_CHARS]
            if len(chunk) < 120:
                continue
            snippets.append(
                {
                    "id": f"case:{path.stem}:{i}",
                    "source_type": "caselaw",
                    "citation": f"{name} {cite}".strip() if name else cite,
                    "aliases": sorted(set(filter(None, [cite, name, short_name]))),
                    "title": name or cite,
                    "text": chunk,
                    "url": f"https://caselaw.nationalarchives.gov.uk/{path.stem.replace('_','/')}",
                }
            )
    return snippets


# --------------------------------------------------------------------------- #
# Guidance (curated PACE Codes reference)
# --------------------------------------------------------------------------- #
def parse_guidance() -> list[dict]:
    snippets: list[dict] = []
    ref_path = GUIDANCE_DIR / "pace_codes_reference.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        for i, prov in enumerate(ref.get("provisions", [])):
            cite = prov["citation"]
            snippets.append(
                {
                    "id": f"guid:pace:{i}",
                    "source_type": "guidance",
                    "citation": cite,
                    "aliases": sorted(set([cite, cite.replace("PACE ", "")])),
                    "title": prov.get("topic", "PACE Code provision"),
                    "text": prov["text"],
                    "url": "https://www.gov.uk/guidance/police-and-criminal-evidence-act-1984-pace-codes-of-practice",
                }
            )
    return snippets


def main() -> None:
    leg = parse_legislation()
    case = parse_caselaw()
    guid = parse_guidance()
    snippets = leg + case + guid
    write_jsonl(CORPUS_INDEX, snippets)
    print(f"[corpus] legislation snippets: {len(leg)}")
    print(f"[corpus] caselaw snippets:     {len(case)}")
    print(f"[corpus] guidance snippets:    {len(guid)}")
    print(f"[corpus] total:                {len(snippets)} -> {CORPUS_INDEX}")


if __name__ == "__main__":
    main()
