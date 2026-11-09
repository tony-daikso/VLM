import re
import numpy as np
ALL_LV = [f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)] + ["S1"]
COV = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
IDX = {l: i for i, l in enumerate(ALL_LV)}
TOK = re.compile(r"\b([CTLS])\s?-?(\d{1,2})\b")
RNG = re.compile(r"\b([CTLS])\s?(\d{1,2})\s*(?:-|–|to|through)\s*([CTLS])?\s?(\d{1,2})\b")
VERT_KW = re.compile(r"compress|wedg|height|endplate|end plate|vertebroplast|kyphoplast|collapse|plana|burst|vertebral body|vertebral bodies|fracture|deformit", re.I)


BONE_S = re.compile(r"bone|osseous|spine|spinal|vertebr|musculoskeletal|skeletal|osteo|scolio|kyphos|schmorl|endplate|vertebral compress|compression deformit|compression fracture|wedg\w* (?:deformit|compress|of [TL])|anterior wedg|fracture|fusion|hardware|laminect|lytic|sclerotic|metasta|degenerat|spondyl|disc", re.I)


def msk_section(rep):
    """bone-related sentences of the whole report (findings + impression); robust to varying section headers."""
    if not isinstance(rep, str):
        return ""
    return " ".join(s for s in sentences(rep) if BONE_S.search(s))


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.;])\s+|\|\|", t) if s.strip()]


def levels_in(s):
    out = set()
    for a, b, c, d in RNG.findall(s):
        c = c or a
        la, lb = f"{a}{int(b)}", f"{c}{int(d)}"
        if la in IDX and lb in IDX and IDX[lb] >= IDX[la] and IDX[lb] - IDX[la] <= 8:
            out.update(ALL_LV[IDX[la]:IDX[lb] + 1])
    for a, b in TOK.findall(s):
        l = f"{a}{int(b)}"
        if l in IDX:
            out.add(l)
    return out


def compression_levels(quotes):
    """levels named in vertebral-fracture-like sentences of the LLM quotes (rib / process sentences dropped)."""
    lv = set()
    for s in sentences(quotes):
        if not VERT_KW.search(s):
            continue
        if re.search(r"\brib|transverse process|spinous process|pars\b|spondylolys|facet|lamina|pedicle", s, re.I) and not re.search(r"compress|wedg|height|endplate|vertebral bod", s, re.I):
            continue
        lv |= levels_in(s)
    return sorted(lv, key=lambda x: IDX[x])


KW = {
    "mild": r"\bmild|minimal|slight|small|subtle|minor|trace|grade 1\b|less than 25|<\s?25",
    "moderate_severe": r"moderate|severe|marked|significant|grade [23]\b|near(ly)? complete|plana|>\s?(40|50)|\b(40|50|60|70|80)\s?%|burst|retropuls",
    "chronic": r"chronic|old|remote|healed|stable|unchanged|again|redemonstrat|similar|prior|known|previously|age.indeterminate|not new",
    "acute": r"\bacute|new\b|recent|subacute|edema|interval development|interval (?:new|increase)",
    "endplate_only": r"endplate|end plate|schmorl|cupping",
    "treated": r"vertebroplast|kyphoplast|cement|augmentation",
    "hardware": r"fusion|rod|screw|hardware|laminectomy|instrumentation|pedicle screw|spacer|cage",
    "posterior_elem": r"transverse process|spinous process|pars\b|spondylolys|lamina|pedicle|facet",
    "hedge": r"possible|possibly|questionable|may represent|versus|vs\.|cannot be excluded|likely|suspect|probabl|developmental|limbus|congenital|not significantly changed",
    "patholog": r"patholog|metasta|lytic|myeloma|lesion",
    "wedge_l5_post": r"posterior wedg",
}
KWC = {k: re.compile(v, re.I) for k, v in KW.items()}
NEG_VERT = re.compile(r"(?<!no )(?<!without )(compression (?:deformit|fracture)|wedg(?:e|ing)|loss of (?:vertebral )?(?:body )?height|height loss|vertebroplast|kyphoplast|schmorl|endplate (?:deformit|depress|irregular|fracture))", re.I)
NEGATED = re.compile(r"\bno (?:acute |evidence of |significant |definite )*(?:\w+ ){0,3}(?:compression|fracture|wedg|height loss)|without (?:\w+ ){0,3}(?:compression|fracture)", re.I)


def flags(text):
    return {k: bool(c.search(text or "")) for k, c in KWC.items()}

VCF = re.compile(r"compression (?:deformit|fracture)|(?:anterior |mild |vertebral )wedg|wedg\w* (?:deformit|compression|of (?:the )?[TL]\d)|(?:loss of|lost) (?:vertebral )?(?:body )?height|height loss|vertebroplast|kyphoplast|endplate (?:deformit|depress|compression|fracture|collapse)|vertebra plana|burst fracture", re.I)
VCTX = re.compile(r"\b[TL]\s?\d{1,2}\b|vertebr|spine|endplate|thoracolumbar|lumbar|thoracic", re.I)
NEGQ = re.compile(r"\b(?:no|without|negative for|not)\b(?:\W+\w+){0,5}\W+(?:compression|wedg|height|fracture|deformit)", re.I)


def vcf_mention(text):
    """non-negated vertebral compression-type phrase in any sentence -> list of short phrases"""
    out = []
    for s in sentences(text or ""):
        if VCF.search(s) and VCTX.search(s) and not NEGQ.search(s):
            out.append(s)
    return out
