"""Deterministic catalogue of England & Wales policing scenario seeds.

build_scenarios(n) returns n distinct seed dicts describing an incident, the
officer's question, and the scene context. Generation (generate_dataset.py)
renders each into a SCENE CARD user message matching format.jsonl and retrieves
grounding snippets using `retrieval_terms`.

No randomness is used (so runs are reproducible); variation comes from index math.
"""
from __future__ import annotations

# Each incident: stable retrieval terms (for BM25) + several officer queries.
INCIDENTS = [
    {
        "key": "ss_drugs",
        "incident_type": "Stop and search — suspected possession of controlled drugs",
        "retrieval_terms": "stop and search controlled drugs reasonable grounds suspicion "
                           "section 23 Misuse of Drugs Act section 1 PACE Code A cannabis",
        "key_facts": "Officer reports a strong smell of cannabis from the subject's clothing. "
                     "Subject has hands in pockets, asks why he has been stopped.",
        "queries": [
            "Do I have grounds to search this person, and what must I tell them first?",
            "The subject is demanding to know why he is being searched. What are my obligations?",
            "Can I search his bag based on the smell of cannabis alone?",
        ],
        "esc": 0.45,
    },
    {
        "key": "ss_weapon",
        "incident_type": "Stop and search — suspected offensive weapon",
        "retrieval_terms": "stop and search offensive weapon bladed article section 1 PACE "
                           "reasonable grounds Code A GOWISELY",
        "key_facts": "A member of the public reported the subject was seen concealing a knife. "
                     "Subject matches the description and is standing still, hands visible.",
        "queries": [
            "What are my legal grounds to search for a weapon here?",
            "Must I give the subject any information before I begin the search?",
            "The subject refuses to be searched. What can I do?",
        ],
        "esc": 0.58,
    },
    {
        "key": "ss_s60",
        "incident_type": "Stop and search — section 60 authorisation in force",
        "retrieval_terms": "section 60 Criminal Justice and Public Order Act 1994 authorisation "
                           "serious violence no reasonable suspicion stop search inspector",
        "key_facts": "A section 60 authorisation is in force in this area following reports of "
                     "gang-related violence. Subject is walking through the zone.",
        "queries": [
            "Under the section 60 authorisation, do I need reasonable suspicion to search him?",
            "What are the limits on my powers under this section 60 authorisation?",
            "The subject says I cannot search him without a reason. How do I respond?",
        ],
        "esc": 0.5,
    },
    {
        "key": "rt_documents",
        "incident_type": "Traffic stop — failure to produce documents",
        "retrieval_terms": "road traffic act 1988 section 163 power to stop section 164 165 "
                           "produce driving licence insurance documents constable",
        "key_facts": "Driver was stopped for a defective brake light. Driver states he is not "
                     "carrying his licence and questions whether he must provide details.",
        "queries": [
            "The driver refuses to give his details. What are my options?",
            "Is the driver legally required to produce his licence at the roadside?",
            "What can I require from the driver and what happens if he refuses?",
        ],
        "esc": 0.4,
    },
    {
        "key": "rt_drink",
        "incident_type": "Traffic stop — suspected drink driving",
        "retrieval_terms": "road traffic act 1988 section 6 preliminary breath test specimen "
                           "drink driving impaired constable require",
        "key_facts": "Officer detects a smell of alcohol on the driver's breath and notes slurred "
                     "speech. Driver admits to 'a couple of pints earlier'.",
        "queries": [
            "Can I require a roadside breath test, and on what basis?",
            "The driver is refusing to provide a breath specimen. What are the consequences?",
            "What is my power to require a preliminary test here?",
        ],
        "esc": 0.47,
    },
    {
        "key": "arrest_necessity",
        "incident_type": "Potential arrest — necessity assessment",
        "retrieval_terms": "section 24 PACE arrest necessity reasonable grounds suspect believe "
                           "Code G Hayes Merseyside name address prompt investigation",
        "key_facts": "Officer has reasonable grounds to suspect the subject committed a minor "
                     "criminal damage offence. Subject has given a name but no fixed address.",
        "queries": [
            "Do I have a lawful basis to arrest, and is arrest necessary here?",
            "What must I satisfy before arresting under section 24?",
            "Is arrest proportionate, or is there an alternative to arrest?",
        ],
        "esc": 0.55,
    },
    {
        "key": "detention_rights",
        "incident_type": "Custody — detainee rights and caution",
        "retrieval_terms": "PACE Code C caution right to legal advice section 58 solicitor "
                           "section 56 someone informed detention questioning interview",
        "key_facts": "Subject has just been brought into custody. He is asking whether he has to "
                     "answer questions and whether he can get a solicitor.",
        "queries": [
            "What rights must I make sure this detainee is told about?",
            "The detainee wants to know if he must answer questions. What should I convey?",
            "When must I caution him and what is the correct caution?",
        ],
        "esc": 0.3,
    },
    {
        "key": "breach_peace",
        "incident_type": "Public disturbance — possible breach of the peace",
        "retrieval_terms": "breach of the peace common law constable prevent imminent harm "
                           "Hicks detention preventive necessary proportionate",
        "key_facts": "Two neighbours are in a heated argument in the street; one has squared up to "
                     "the other. No blows have been struck. A small crowd is gathering.",
        "queries": [
            "Can I detain someone to prevent a breach of the peace here?",
            "What is my power if I believe a breach of the peace is imminent?",
            "How long can I hold someone to prevent further disorder?",
        ],
        "esc": 0.6,
    },
    {
        "key": "public_order",
        "incident_type": "Public order — threatening or abusive behaviour",
        "retrieval_terms": "public order act 1986 section 5 section 4 harassment alarm distress "
                           "threatening abusive section 14 conditions assembly",
        "key_facts": "Subject is shouting abuse at passers-by outside a pub, using threatening "
                     "language. Several members of the public appear alarmed.",
        "queries": [
            "What offence might apply and what should I consider before acting?",
            "Can I deal with this by way of section 5 Public Order Act?",
            "What are my options to manage this without escalating?",
        ],
        "esc": 0.57,
    },
    {
        "key": "use_of_force",
        "incident_type": "Use of force — proportionality assessment",
        "retrieval_terms": "use of force section 3 Criminal Law Act 1967 reasonable proportionate "
                           "section 117 PACE human rights article 3 ZH disability necessary",
        "key_facts": "A subject is resisting being handcuffed after a lawful detention. He appears "
                     "distressed and may have a vulnerability. He is not striking out.",
        "queries": [
            "How much force may I lawfully use to handcuff a resisting subject?",
            "The subject appears vulnerable. How does that affect my use of force?",
            "What is the legal test for the force I am about to use?",
        ],
        "esc": 0.62,
    },
    {
        "key": "filming",
        "incident_type": "Member of public filming officers",
        "retrieval_terms": "filming photographing police public Wood Commissioner article 8 "
                           "no power to seize phone delete footage data privacy",
        "key_facts": "A bystander is filming the incident on a phone and standing a few metres "
                     "back. The bystander is not obstructing officers.",
        "queries": [
            "Can I require the bystander to stop filming or hand over the phone?",
            "Do I have any power to seize or delete the footage?",
            "The bystander is filming. What are my powers and limits?",
        ],
        "esc": 0.35,
    },
    {
        "key": "mental_health",
        "incident_type": "Person in mental health crisis in a public place",
        "retrieval_terms": "mental health act 1983 section 136 place of safety constable removal "
                           "immediate need care control vulnerable crisis",
        "key_facts": "Subject is standing on a bridge parapet, distressed, and appears to be in a "
                     "mental health crisis. He is in a public place.",
        "queries": [
            "What is my power to take this person to a place of safety?",
            "Can I detain him under the Mental Health Act here?",
            "What should I prioritise given his apparent vulnerability?",
        ],
        "esc": 0.68,
    },
    {
        "key": "entry_search",
        "incident_type": "Entry and search of premises",
        "retrieval_terms": "PACE section 17 section 18 section 32 entry search premises "
                           "without warrant arrest evidence constable power",
        "key_facts": "Officer has just arrested a subject outside a house for a burglary offence "
                     "and wishes to search the premises he occupies.",
        "queries": [
            "Do I have a power to search the premises following this arrest?",
            "On what basis may I enter and search without a warrant?",
            "What are the limits of a search under section 18?",
        ],
        "esc": 0.42,
    },
    {
        "key": "vehicle_cannabis",
        "incident_type": "Traffic stop — smell of cannabis from vehicle",
        "retrieval_terms": "stop search vehicle cannabis smell section 23 Misuse of Drugs Act "
                           "road traffic section 163 reasonable grounds Code A",
        "key_facts": "Vehicle stopped for speeding. Officer notes a strong smell of cannabis from "
                     "the open window. Driver is cooperative; passenger is nervous.",
        "queries": [
            "Can I search the vehicle based on the smell of cannabis?",
            "What grounds do I have to search the car and occupants?",
            "How should I handle the search of the vehicle and the occupants' rights?",
        ],
        "esc": 0.44,
    },
]

LOCATIONS = [
    "High Street, Camden, London",
    "A34 layby near Newbury, Berkshire",
    "Bull Ring, Birmingham",
    "Briggate, Leeds city centre",
    "Queen Street, Cardiff",
    "Lord Street, Liverpool",
    "Gallowgate, Newcastle upon Tyne",
    "Park Street, Bristol",
    "St Peter's Street, Manchester",
    "London Road, Leicester",
]

OFFICERS = [
    ("PC Aservda", 2, "Personal Safety Training"),
    ("PC Okafor", 5, "PACE-trained, Drugs Recognition"),
    ("PS Whitfield", 11, "Public Order Level 2, Personal Safety Training"),
    ("PC Doyle", 3, "Mental Health First Aid, Personal Safety Training"),
    ("PC Rahman", 7, "Roads Policing, Evidential Breath Testing"),
    ("PC Mensah", 1, "Student officer, tutor-supervised"),
    ("PS Calloway", 14, "Custody-trained, PACE-trained"),
]

WEATHERS = [
    "Night, clear, 9°C", "Daytime, overcast, 14°C", "Evening, light rain, 11°C",
    "Daytime, sunny, 19°C", "Night, drizzle, 7°C", "Morning, foggy, 5°C",
]

SUBJECTS = [
    "1 male, ~20s, agitated, raised voice once",
    "1 male, ~30s, calm but uncooperative, hands visible",
    "1 female, ~40s, anxious, repeatedly reaching into pockets",
    "1 male, ~50s, cooperative, answering questions",
    "1 male, ~teens, nervous, glancing around",
    "2 subjects, both male ~20s; one agitated, one passive",
]

DURATIONS = ["1m 05s", "2m 40s", "3m 15s", "4m 30s", "5m 50s", "0m 45s"]


def _label(esc: float) -> str:
    if esc < 0.35:
        return "LOW"
    if esc < 0.65:
        return "ELEVATED"
    return "HIGH"


def build_scenarios(n: int = 100) -> list[dict]:
    seeds: list[dict] = []
    i = 0
    # Round-robin over incidents x their queries, perturbing context by index.
    rounds = 0
    while len(seeds) < n:
        for inc in INCIDENTS:
            if len(seeds) >= n:
                break
            q = inc["queries"][rounds % len(inc["queries"])]
            # Deterministic escalation perturbation per round.
            esc = round(min(0.92, max(0.18, inc["esc"] + (((i % 5) - 2) * 0.05))), 2)
            hh = 6 + (i * 7) % 18
            mm = (i * 13) % 60
            ss = (i * 29) % 60
            seeds.append(
                {
                    "id": f"{inc['key']}_{rounds}",
                    "incident_type": inc["incident_type"],
                    "retrieval_terms": inc["retrieval_terms"],
                    "query": q,
                    "key_facts": inc["key_facts"],
                    "location": LOCATIONS[i % len(LOCATIONS)],
                    "subjects": SUBJECTS[i % len(SUBJECTS)],
                    "officer": OFFICERS[i % len(OFFICERS)],
                    "weather": WEATHERS[i % len(WEATHERS)],
                    "duration": DURATIONS[i % len(DURATIONS)],
                    "escalation_index": esc,
                    "escalation_label": _label(esc),
                    "time": f"{hh:02d}:{mm:02d}:{ss:02d}",
                    "jurisdiction": "England & Wales",
                    "prior_incidents": "None" if i % 3 else "1 (verbal warning, 6 months prior)",
                }
            )
            i += 1
        rounds += 1
    return seeds[:n]


if __name__ == "__main__":
    s = build_scenarios(100)
    print(f"{len(s)} scenarios across {len(INCIDENTS)} incident types")
    from collections import Counter

    print(Counter(x["id"].rsplit("_", 1)[0] for x in s))
