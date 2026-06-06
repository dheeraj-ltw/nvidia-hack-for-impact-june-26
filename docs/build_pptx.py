"""Generate docs/PatrolAssist.pptx — the hackathon pitch deck.

Clean light / NVIDIA theme. Mirrors docs/PRESENTATION.md.
Re-run after editing content:
    .venv/bin/python docs/build_pptx.py
"""
from __future__ import annotations

import pathlib

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

DOCS = pathlib.Path(__file__).resolve().parent
ARCH_PNG = DOCS / "architecture_light.png"
LOGO_PNG = DOCS / "jarvis-logo.png"   # rsvg-convert -w 600 jarvis-logo.svg -o docs/jarvis-logo.png
OUT = DOCS / "JARVIS.pptx"

# ---- Palette: clean light, NVIDIA green ----------------------------------
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1C, 0x21, 0x2E)   # charcoal headlines/body
INK_SOFT = RGBColor(0x3B, 0x42, 0x52)  # secondary body
MUTED = RGBColor(0x6B, 0x72, 0x80)  # subtitles / captions
GREEN = RGBColor(0x76, 0xB9, 0x00)  # NVIDIA green (accents, bars, table head)
GREEN_TX = RGBColor(0x4E, 0x7A, 0x00)  # darker green for small text on white (contrast)
HAIR = RGBColor(0xE3, 0xE7, 0xEC)   # hairline rules / borders
PANEL = RGBColor(0xF5, 0xF7, 0xF1)  # quote / alt-row panel (faint green-grey)

FOOTER = "JARVIS  ·  NVIDIA Impact Hackathon"

W, H = Inches(13.333), Inches(7.5)
ML = Inches(0.85)          # left margin
MR = Inches(0.85)          # right margin
CW = W - ML - MR           # content width

prs = Presentation()
prs.slide_width = W
prs.slide_height = H
BLANK = prs.slide_layouts[6]

_page = 0


# ----------------------------------------------------------------- helpers
def _bg(slide):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = WHITE


def _no_line(shape):
    shape.line.fill.background()


def _rect(slide, l, t, w, h, color, shape=MSO_SHAPE.RECTANGLE):
    sp = slide.shapes.add_shape(shape, l, t, w, h)
    sp.fill.solid()
    sp.fill.fore_color.rgb = color
    _no_line(sp)
    sp.shadow.inherit = False
    return sp


def _box(slide, l, t, w, h):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    return tb, tf


def _run(p, text, size, color=INK, bold=False, italic=False, font="Calibri"):
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = font
    return r


def _link(p, text, url, size, color=GREEN_TX, bold=True):
    r = _run(p, text, size, color, bold=bold)
    r.font.underline = True
    r.hyperlink.address = url
    # hyperlink can reset run colour to theme; re-assert ours
    r.font.color.rgb = color
    return r


def _chrome(slide):
    """Footer rule + running title + page number on a content slide."""
    global _page
    _page += 1
    # footer hairline
    _rect(slide, ML, Inches(6.95), CW, Pt(1), HAIR)
    _, lf = _box(slide, ML, Inches(7.04), Inches(9.0), Inches(0.35))
    _run(lf.paragraphs[0], FOOTER, 9, MUTED)
    _, rf = _box(slide, W - MR - Inches(1.2), Inches(7.04), Inches(1.2), Inches(0.35))
    rp = rf.paragraphs[0]
    rp.alignment = PP_ALIGN.RIGHT
    _run(rp, f"{_page:02d}", 9, GREEN_TX, bold=True)


def _header(slide, title, kicker=None):
    """Left accent bar + kicker tag + title + green rule."""
    # accent bar down the left edge
    _rect(slide, Inches(0.0), Inches(0.0), Inches(0.16), H, GREEN)
    top = Inches(0.55)
    if kicker:
        _, kf = _box(slide, ML, top, CW, Inches(0.3))
        _run(kf.paragraphs[0], kicker.upper(), 12, GREEN_TX, bold=True)
        top = Inches(0.92)
    _, tt = _box(slide, ML, top, CW, Inches(0.7))
    _run(tt.paragraphs[0], title, 30, INK, bold=True)
    rule_t = top + Inches(0.74)
    _rect(slide, ML, rule_t, Inches(1.6), Pt(3), GREEN)
    return rule_t + Inches(0.22)   # y where body can start


# ----------------------------------------------------------------- layouts
def title_slide(kicker, headline, note):
    """Logo as the hero (it carries the JARVIS wordmark + tagline) + pitch copy."""
    s = prs.slides.add_slide(BLANK)
    _bg(s)
    _rect(s, 0, 0, Inches(0.45), H, GREEN)  # left accent column
    if LOGO_PNG.exists():
        s.shapes.add_picture(str(LOGO_PNG), Inches(1.05), Inches(1.35), height=Inches(4.8))
    else:
        _, jf = _box(s, Inches(1.1), Inches(3.0), Inches(4), Inches(1.2))
        _run(jf.paragraphs[0], "JARVIS", 52, INK, bold=True)
    tx, tw = Inches(5.75), Inches(6.85)
    _, kf = _box(s, tx, Inches(2.3), tw, Inches(0.4))
    _run(kf.paragraphs[0], kicker.upper(), 14, GREEN_TX, bold=True)
    _, hf = _box(s, tx, Inches(2.85), tw, Inches(1.8))
    _run(hf.paragraphs[0], headline, 27, INK, bold=True)
    _rect(s, tx, Inches(4.7), Inches(2.0), Pt(4), GREEN)
    if note:
        _, nf = _box(s, tx, Inches(4.95), tw, Inches(1.0))
        _run(nf.paragraphs[0], note, 15, MUTED, italic=True)
    return s


def content_slide(title, blocks, kicker=None):
    """blocks: (text, level, kind) — kind in {head, bullet, sub, quote}."""
    s = prs.slides.add_slide(BLANK)
    _bg(s)
    body_top = _header(s, title, kicker)
    _, tf = _box(s, ML, body_top, CW, Inches(6.7) - body_top)
    first = True
    for text, level, kind in blocks:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = 0
        if kind == "head":
            p.space_before = Pt(13)
            p.space_after = Pt(2)
            _run(p, text, 19, GREEN_TX, bold=True)
        elif kind == "quote":
            p.space_before = Pt(10)
            _run(p, text, 16, INK_SOFT, italic=True)
        elif kind == "sub":
            p.space_before = Pt(2)
            _run(p, text, 14, MUTED)
        else:  # bullet
            p.space_before = Pt(7)
            indent = "    " if level else ""
            mark = "–  " if level else "•  "
            _run(p, indent + mark, 17, GREEN, bold=True)
            _run(p, text, 17, INK)
    _chrome(s)
    return s


def rich_slide(title, segments, kicker=None):
    """segments: list of paragraphs; each a list of (text,color,bold,italic,size)."""
    s = prs.slides.add_slide(BLANK)
    _bg(s)
    body_top = _header(s, title, kicker)
    _, tf = _box(s, ML, body_top + Inches(0.2), CW, Inches(6.5) - body_top)
    for i, runs in enumerate(segments):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(14)
        for (text, color, bold, italic, size) in runs:
            _run(p, text, size, color, bold=bold, italic=italic)
    _chrome(s)
    return s


def _set_cell(cell, text, size, color, bold=False, fill=None, align=PP_ALIGN.LEFT):
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Inches(0.14)
    cell.margin_right = Inches(0.1)
    cell.margin_top = Inches(0.04)
    cell.margin_bottom = Inches(0.04)
    if fill is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
    else:
        cell.fill.background()
    para = cell.text_frame.paragraphs[0]
    para.alignment = align
    _run(para, text, size, color, bold=bold)


def _strip_table_style(table):
    """Remove banded-row theme styling so our explicit fills show cleanly."""
    tbl = table._tbl
    pr = tbl.find(qn("a:tblPr"))
    if pr is not None:
        pr.set("firstRow", "0")
        pr.set("bandRow", "0")
        for child in list(pr):
            if child.tag == qn("a:tableStyleId"):
                pr.remove(child)


def table_slide(title, headers, rows, col_widths=None, intro=None, kicker=None):
    s = prs.slides.add_slide(BLANK)
    _bg(s)
    body_top = _header(s, title, kicker)
    if intro:
        _, itf = _box(s, ML, body_top, CW, Inches(0.5))
        _run(itf.paragraphs[0], intro, 16, GREEN_TX, bold=True)
        body_top = body_top + Inches(0.55)
    nrows, ncols = len(rows) + 1, len(headers)
    gframe = s.shapes.add_table(nrows, ncols, ML, body_top, CW, Inches(0.4))
    table = gframe.table
    _strip_table_style(table)
    if col_widths:
        for c, w in enumerate(col_widths):
            table.columns[c].width = Inches(w)
    table.rows[0].height = Inches(0.5)
    for c, htext in enumerate(headers):
        _set_cell(table.cell(0, c), htext, 15, WHITE, bold=True, fill=GREEN)
    for r, row in enumerate(rows, start=1):
        table.rows[r].height = Inches(0.52)
        fill = PANEL if r % 2 else WHITE
        for c, val in enumerate(row):
            last = c == ncols - 1
            color = GREEN_TX if (last and ("Built" in val or "✅" in val or "🔜" in val)) else INK
            _set_cell(table.cell(r, c), val, 14, color, bold=last and "✅" in val, fill=fill)
    _chrome(s)
    return s


def image_slide(title, img, caption=None, kicker=None):
    s = prs.slides.add_slide(BLANK)
    _bg(s)
    body_top = _header(s, title, kicker)
    if img.exists():
        pic = s.shapes.add_picture(str(img), ML, body_top, height=Inches(4.3))
        pic.left = int((W - pic.width) / 2)
    else:
        _, tf = _box(s, ML, Inches(3), CW, Inches(1))
        _run(tf.paragraphs[0], f"[ {img.name} not found ]", 16, MUTED, italic=True)
    if caption:
        _, cf = _box(s, ML, Inches(6.25), CW, Inches(0.6))
        cf.paragraphs[0].alignment = PP_ALIGN.CENTER
        _run(cf.paragraphs[0], caption, 14, MUTED, italic=True)
    _chrome(s)
    return s


SOURCES = [
    ("legislation.gov.uk", "https://www.legislation.gov.uk"),
    ("Find Case Law (The National Archives)", "https://caselaw.nationalarchives.gov.uk"),
    ("PACE Codes", "https://www.gov.uk/guidance/police-and-criminal-evidence-act-1984-pace-codes-of-practice"),
]


def technical_depth_slide():
    """Custom layout so the data sources can be rendered as clickable hyperlinks."""
    s = prs.slides.add_slide(BLANK)
    _bg(s)
    bt = _header(s, "Technical depth — a system, not a wrapper",
                 kicker="Execution & completeness · 30 pts")
    _, tf = _box(s, ML, bt, CW, Inches(6.7) - bt)

    def head(text, first=False, gap=12):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.space_before = Pt(0 if first else gap)
        _run(p, text, 18, GREEN_TX, bold=True)
        return p

    def bullet(text):
        p = tf.add_paragraph()
        p.space_before = Pt(3)
        _run(p, "•  ", 16, GREEN, bold=True)
        _run(p, text, 16, INK)
        return p

    head("1 · Legal-reasoning dataset pipeline  (built)", first=True)
    src = tf.add_paragraph()
    src.space_before = Pt(3)
    _run(src, "Built on real, openly-licensed England & Wales law:  ", 16, INK)
    for i, (label, url) in enumerate(SOURCES):
        if i:
            _run(src, "  ·  ", 15, MUTED)
        _link(src, label, url, 15)
    bullet("citation-tagged corpus → BM25 retrieval → SCENE-CARD examples (<think> reasoning + "
           "structured JSON) → validate. Every citation is cross-checked back against the corpus.")

    head("2 · Fine-tuned an NVIDIA model  (trained)")
    bullet("QLoRA on Nemotron-Super-49B → merged → GGUF Q4_K_M for local serving.")

    head("3 · Real-time orchestrator  (built)")
    bullet("Throttled drop-if-busy loop (≤1 analysis / 0.7 s) keeps latency bounded under load.")

    head("4 · Local inference services  (built)")
    bullet("Self-hosted vLLM (video) + llama.cpp (the fine-tuned LLM), both OpenAI-compatible, "
           "containerized.")
    _chrome(s)
    return s


def value_impact_slide():
    """Bespoke layout: the example rendered as a structured, attributed guidance card."""
    s = prs.slides.add_slide(BLANK)
    _bg(s)
    bt = _header(s, "Value & impact", kicker="Value & impact · 20 pts")

    # intro
    _, itf = _box(s, ML, bt, CW, Inches(0.8))
    _run(itf.paragraphs[0], "Insight quality — situation-specific, not generic.", 18, GREEN_TX, bold=True)
    p2 = itf.add_paragraph()
    p2.space_before = Pt(3)
    _run(p2, "A dashboard says ", 15, INK)
    _run(p2, "“crime rises at night.”", 15, MUTED, italic=True)
    _run(p2, "  JARVIS reasons about the actual scene and cites the law:", 15, INK)

    # guidance card
    rows = [
        ("Scene", "strong smell of cannabis · hands in pockets · 2-yr officer · night"),
        ("Legal basis", "s.23(2) Misuse of Drugs Act 1971 — reasonable grounds to search"),
        ("Say first", "all GOWISELY elements (PACE Code A 3.8)"),
        ("Rights", "street search, not arrest — Code C custody rights not yet engaged"),
        ("De-escalation", "answer “why am I stopped?” directly; speak slowly"),
    ]
    card_top = bt + Inches(1.02)
    row_h = Inches(0.34)
    card_h = row_h * len(rows) + Inches(0.24)
    _rect(s, ML, card_top, CW, card_h, PANEL)
    _rect(s, ML, card_top, Inches(0.09), card_h, GREEN)  # card accent edge
    _, ctf = _box(s, ML + Inches(0.3), card_top + Inches(0.12), CW - Inches(0.5), card_h - Inches(0.24))
    for i, (label, value) in enumerate(rows):
        p = ctf.paragraphs[0] if i == 0 else ctf.add_paragraph()
        p.space_before = Pt(0 if i == 0 else 6)
        lr = _run(p, f"{label}", 14, GREEN_TX, bold=True)
        _run(p, "      ", 14, INK)
        _run(p, value, 14, INK)

    # attribution — directly counters "this looks hallucinated"
    att_top = card_top + card_h + Inches(0.1)
    _, atf = _box(s, ML, att_top, CW, Inches(0.35))
    _run(atf.paragraphs[0],
         "Actual model output — every citation is validated back against the scraped corpus, not invented.",
         13, MUTED, italic=True)

    # usability
    use_top = att_top + Inches(0.5)
    _, utf = _box(s, ML, use_top, CW, Inches(1.3))
    _run(utf.paragraphs[0], "Usability — a tool an officer could use tomorrow.", 18, GREEN_TX, bold=True)
    b1 = utf.add_paragraph()
    b1.space_before = Pt(6)
    _run(b1, "•  ", 16, GREEN, bold=True)
    _run(b1, "Officer: hands-free audio guidance, human-in-the-loop — nothing decided for them.", 16, INK)
    b2 = utf.add_paragraph()
    b2.space_before = Pt(6)
    _run(b2, "•  ", 16, GREEN, bold=True)
    _run(b2, "Supervisor / legal reviewer: replayable MP4 + synced transcript + guidance log per session.",
         16, INK)
    _chrome(s)
    return s


def section_slide(kicker, title, subtitle):
    s = prs.slides.add_slide(BLANK)
    _bg(s)
    _rect(s, 0, 0, W, H, PANEL)
    _rect(s, 0, 0, Inches(0.45), H, GREEN)
    if LOGO_PNG.exists():
        s.shapes.add_picture(str(LOGO_PNG), W - Inches(2.55), Inches(0.7), height=Inches(2.0))
    _, kf = _box(s, Inches(1.1), Inches(2.9), Inches(11), Inches(0.4))
    _run(kf.paragraphs[0], kicker.upper(), 14, GREEN_TX, bold=True)
    _, tf = _box(s, Inches(1.1), Inches(3.35), Inches(11), Inches(1.1))
    _run(tf.paragraphs[0], title, 44, INK, bold=True)
    _rect(s, Inches(1.13), Inches(4.5), Inches(2.0), Pt(4), GREEN)
    if subtitle:
        _, sf = _box(s, Inches(1.1), Inches(4.75), Inches(11), Inches(0.6))
        _run(sf.paragraphs[0], subtitle, 18, MUTED)
    return s


# ---------------------------------------------------------------- build deck
title_slide(
    "NVIDIA Impact Hackathon  ·  runs locally on DGX Spark",
    "A real-time, on-device legal & de-escalation copilot for frontline officers",
    "Assistive and human-in-the-loop. It cites the law; it never decides.",
)

content_slide("The problem", [
    ("Officers make split-second legal decisions under stress:", 0, "head"),
    ("Do I have grounds to search? What must I say first?", 0, "bullet"),
    ("Is arrest necessary, or is there a lesser option?", 0, "bullet"),
    ("Which rights are engaged right now?", 0, "bullet"),
    ("Get it wrong and the cost is real: unlawful stops, avoidable escalation, "
     "evidence thrown out, lost public trust.", 0, "head"),
    ("Today, body-worn video is review-only — it tells you what went wrong after the "
     "encounter. There is no real-time copilot that reads the scene and the law in the moment.",
     0, "quote"),
], kicker="The gap")

content_slide("What we built", [
    ("A live body-cam feed (video + audio) streams to a local pipeline that:", 0, "head"),
    ("Sees the scene — vision model describes people, objects, actions", 0, "bullet"),
    ("Hears the conversation — speech-to-text, with who-said-what (officer vs. subject)", 0, "bullet"),
    ("Reasons over real England & Wales law — cited, de-escalation-focused guidance", 0, "bullet"),
    ("Speaks it back — hands-free audio for the officer", 0, "bullet"),
    ("Records everything — tamper-evident audit log + replayable MP4 with synced transcript", 0, "bullet"),
    ("Built to run entirely on a DGX Spark — body-cam footage and PII never leave the device.",
     0, "head"),
    ("JARVIS = Patrol Assist (the real-time system) + PoliceAI (the fine-tuned, law-grounded model).",
     0, "sub"),
], kicker="Solution")

content_slide("Demo flow", [
    ("Onboard the officer — record a ~7 s voice sample (enrolls a local voice embedding)", 0, "bullet"),
    ("Start patrol — camera + mic stream over a binary WebSocket at ~2 fps", 0, "bullet"),
    ("Live detections + transcript appear; transcript is tagged officer / subject in real time", 0, "bullet"),
    ("A cited guidance card surfaces (legal basis · action · tone · rights · next steps) and is read aloud",
     0, "bullet"),
    ("Stop — the session is encoded to MP4 and stored; a background pass diarizes the audio for playback",
     0, "bullet"),
    ("Every input, model output, and officer action is written to an append-only session manifest.",
     0, "quote"),
], kicker="In 5 steps")

image_slide(
    "Architecture",
    ARCH_PNG,
    "Field device → Next.js console → FastAPI orchestrator → pluggable AIService layer → reactive UI + TTS, "
    "every session persisted to MinIO/S3.",
    kicker="System",
)

technical_depth_slide()

table_slide(
    "NVIDIA ecosystem & the Spark story",
    ["NVIDIA piece", "How we use it"],
    [
        ["Nemotron-Super-49B", "Fine-tuned with QLoRA into PoliceAI — reasoning + structured output"],
        ["DGX Spark (GB10 Grace Blackwell)", "Training + inference target; 128 GB unified memory"],
        ["FP4 on 5th-gen Tensor Cores", "Native 4-bit → fine-tune a 49B model in ~38 GB"],
        ["NVIDIA vLLM container (nvcr.io)", "Serves the local vision model"],
        ["NIM / Nemotron API", "Pluggable cloud reasoning backend"],
    ],
    col_widths=[4.3, 7.33],
    intro="We did not just call a cloud LLM — we fine-tuned and serve an NVIDIA model locally.",
    kicker="NVIDIA stack · 30 pts",
)

rich_slide("Why Spark, specifically", [
    [("Unified memory.  ", GREEN_TX, True, False, 21),
     ("128 GB holds the 49B model + KV cache + video-frame buffer + speaker embeddings "
      "simultaneously — no sharding, no juggling a 24 GB discrete GPU.", INK, False, False, 19)],
    [("Privacy.  ", GREEN_TX, True, False, 21),
     ("Local inference means evidence-grade body-cam data and PII stay on the device — "
      "essential for policing and the chain of custody.", INK, False, False, 19)],
    [("Latency.  ", GREEN_TX, True, False, 21),
     ("On-device guidance during a live encounter — no round-trip to a cloud API.", INK, False, False, 19)],
    [('"Merely calling GPT-4 via API gets 0 points." We fine-tuned an NVIDIA model and run it locally.',
      MUTED, False, True, 16)],
], kicker="The Spark story · 15 pts")

value_impact_slide()

content_slide("Innovation & execution", [
    ("Creativity — combine modalities that aren't usually combined:", 0, "head"),
    ("Vision (scene) + speaker diarization (officer vs. subject) + legal RAG reasoning + voice — "
     "reads the scene and the law together, in real time.", 0, "bullet"),
    ("Anti-hallucination by construction: trained to cite only retrieved law; a validator drops any "
     "record whose citations don't trace back to the corpus.", 0, "bullet"),
    ("Performance & engineering:", 0, "head"),
    ("Drop-if-busy throttling → bounded real-time latency.", 0, "bullet"),
    ("FP16 QLoRA → 49B fine-tune fits in ~98 GB; quantized GGUF served via llama.cpp.", 0, "bullet"),
    ("Local resemblyzer voice embeddings → speaker labels with no API call.", 0, "bullet"),
], kicker="Innovation · 20 pts")

content_slide("Roadmap & the ask", [
    ("Beyond the hack:", 0, "head"),
    ("Expand corpus + dataset (more incident types, force-policy packs).", 0, "bullet"),
    ("Per-force fine-tunes; on-device continuous learning from reviewed sessions.", 0, "bullet"),
    ("Field trial with a human-in-the-loop supervisor dashboard.", 0, "bullet"),
    ("The ask: a DGX Spark in the loop turns this from a privacy-respecting prototype into "
     "something a real patrol could use tomorrow.", 0, "head"),
    ("JARVIS — it reads the scene, cites the law, and keeps the human in command.", 0, "quote"),
], kicker="What's next")

section_slide("Appendix", "Technical detail", "For Q&A")

content_slide("Appendix · Data + fine-tuning pipeline", [
    ("Sources (openly licensed): legislation.gov.uk (CLML XML), Find Case Law / The National "
     "Archives (LegalDocML), curated PACE Codes A/C/G. Statutes incl. PACE 1984, MDA 1971, "
     "CJPOA 1994 (s.60), RTA 1988, POA 1986, HRA 1998, MHA 1983.", 0, "bullet"),
    ("Generation: 16 incident types × officer queries → deterministic SCENE CARDs → BM25 retrieval "
     "(top-8, with a guidance floor that always keeps PACE-Code slots) → Claude produces <think> + 6-key JSON.",
     0, "bullet"),
    ("Validation (hard): exact system prompt · <think> present · JSON schema · no US law · no guilt "
     "opinion · citation grounding · dedupe → 98 records (88 train / 10 val).", 0, "bullet"),
    ("Fine-tune: Nemotron-Super-49B · QLoRA (r=16, α=32, FP4 base + BF16 adapters) · 3 epochs · "
     "LR 2e-4 · seq 4096 · transformers / trl / peft / bitsandbytes · merge → GGUF Q4_K_M.", 0, "bullet"),
], kicker="Appendix")

content_slide("Appendix · Real-time system internals", [
    ("Capture (browser): ~2 fps JPEG (q0.6); audio as 1 s continuous fragments + 4 s STT clips; "
     "binary WebSocket (1-byte kind + timestamp + dims + payload).", 0, "bullet"),
    ("Orchestrator: analyze at most once / 0.7 s, drop-if-busy; keeps last 2,000 chars of transcript "
     "as reasoning context; emits Detection / Transcript / Guidance / Speech events.", 0, "bullet"),
    ("Speaker ID: local resemblyzer d-vectors, 0.70 cosine threshold (live); post-session ElevenLabs "
     "Scribe diarization relabels officer / person1 / person2…", 0, "bullet"),
    ("Recording: frames + audio + events → MinIO; ffmpeg concat → H.264/AAC MP4; manifest drives playback.",
     0, "bullet"),
    ("Local services: llava-video-to-text — vLLM serving LLaVA-1.6-Mistral-7B (8900); police-llm — "
     "llama.cpp serving the fine-tuned Nemotron GGUF (8000). src/video_to_text.py — remote Qwen2.5-VL "
     "(Nebius) dev/fallback path.", 0, "bullet"),
], kicker="Appendix")

prs.save(str(OUT))
print(f"wrote {OUT}  ({len(prs.slides._sldIdLst)} slides)")
