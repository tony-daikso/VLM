"""Report-text parsing for rib / pelvic fracture descriptions (applied only to train+val rows)."""
import re
ORDW = {w: i + 1 for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split())}
NUM = r"(?:(\d{1,2})(?:st|nd|rd|th)?|(" + "|".join(ORDW) + r"))"
SENT = re.compile(r"(?<=[.;])\s+|\n+")
FRX = re.compile(r"fractur|deformit|callus|\bfx\b|healed|healing|break", re.I)
NEGP = re.compile(r"\bno (?:acute |definite |displaced |evidence of |visible |)(?:acute )?(?:osseous |bony |rib |)fractur|without (?:acute |)fractur|negative for fractur", re.I)
SIDE = re.compile(r"\b(left|right|bilateral|both)\b", re.I)

def sentences(t):
    return [s.strip() for s in SENT.split(t or "") if s.strip()]

def _n(m):
    return int(m.group(1)) if m.group(1) else ORDW[m.group(2).lower()]

def rib_levels(s):
    """-> list of (side, level) from one sentence. side in {L,R,B,?}."""
    out = []
    sl = s.lower()
    # candidate numbers: ordinal-suffixed or ordinal words, or 'rib(s) N', and ranges between them
    pat = re.compile(r"(?<![A-Za-z\d.])" + NUM + r"(?:\s*(?:-|–|to|through|thru)\s*" + NUM + r")?(?=\W)", re.I)
    for m in pat.finditer(s):
        txt = m.group(0)
        a = int(m.group(1)) if m.group(1) else ORDW[m.group(2).lower()]
        has_ord = bool(re.search(r"\d(st|nd|rd|th)|[a-z]", txt, re.I))
        prev = sl[max(0, m.start() - 8):m.start()]
        nxt = sl[m.end():m.end() + 25]
        if not (has_ord or "rib" in prev):
            continue
        if re.match(r"\s*(cm|mm|%|x\b|\.\d|/)", nxt) or sl[max(0, m.start() - 1):m.start()] == "/" or re.search(r"(image|series|img|se)\s*:?\s*$", prev):
            continue
        if not ("rib" in nxt or "rib" in prev or re.match(r"\s*(,|and|or|-|through|to)", nxt)):
            continue
        b = a
        if m.group(3) or m.group(4):
            b = int(m.group(3)) if m.group(3) else ORDW[m.group(4).lower()]
        if not (1 <= a <= 12 and 1 <= b <= 12 and a <= b):
            continue
        sides = [x.group(1).lower() for x in SIDE.finditer(sl[:m.start()])]
        if not sides:
            sides = [x.group(1).lower() for x in SIDE.finditer(sl[m.end():])][:1]
        sd = {"left": "L", "right": "R", "bilateral": "B", "both": "B"}.get(sides[-1] if sides else "", "?")
        out += [(sd, k) for k in range(a, b + 1)]
    # chain like '7th, 8th, and 9th' captured individually; dedupe
    return sorted(set(out))

def rib_info(report):
    ss = [s for s in sentences(report) if re.search(r"\bribs?\b|costal|costovertebral", s, re.I) and FRX.search(s)]
    pos = [s for s in ss if not NEGP.search(s) or re.search(r"\b(old|healed|healing|remote|chronic|prior)\b.*\bribs?\b|\bribs?\b.*\b(old|healed|healing|remote|chronic|prior)\b", s, re.I)]
    t = " ".join(pos).lower()
    lv = sorted(set(x for s in pos for x in rib_levels(s)))
    sides = set(x.group(1) for x in SIDE.finditer(t))
    return dict(
        rib_sent=" | ".join(pos)[:400], n_rib_sent=len(pos),
        rib_levels=";".join(f"{a}{b}" for a, b in lv), n_levels=len(set(b for _, b in lv)),
        lv_min=min((b for _, b in lv), default=None), lv_max=max((b for _, b in lv), default=None),
        side=("B" if ("bilateral" in sides or "both" in sides or {"left", "right"} <= sides) else
              "L" if "left" in sides else "R" if "right" in sides else "?"),
        t_acute=bool(re.search(r"(?<!sub)\bacute\b|\bnew\b|\brecent\b", t) and not re.search(r"no acute", t)),
        t_subacute=bool(re.search(r"subacute", t)),
        t_old=bool(re.search(r"\bold\b|healed|remote|chronic|callus|\bprior\b|post-?traumatic deformit|deformit|sclero", t)),
        t_healing=bool(re.search(r"healing|callus|subacute", t)),
        t_nondisp=bool(re.search(r"non-?displaced|minimally displaced|nondisplaced|subtle|hairline", t)),
        t_disp=bool(re.search(r"(?<!non)(?<!non-)(?<!minimally )\bdisplaced|comminuted|segmental|flail", t)),
        t_multi=bool(re.search(r"multiple|several|numerous|\bribs\b|fractures|bilateral", t)),
        t_ant=bool(re.search(r"anterior|anterolateral|costochondral", t)),
        t_lat=bool(re.search(r"lateral|posterolateral|anterolateral", t)),
        t_post=bool(re.search(r"posterior|posterolateral|costovertebral|paraspinal", t)),
        t_lower=bool(re.search(r"\blower\b|inferior", t)), t_upper=bool(re.search(r"\bupper\b|superior", t)),
        t_uncert=bool(re.search(r"possibl|probabl|may represent|cannot be excluded|not excluded|questionable|suspect|suspicious|equivocal|likely|could represent|versus|vs\.?\b|indeterminate", t)),
        t_patho=bool(re.search(r"patholog|metasta|lytic|lesion|myeloma|insufficiency", t)),
    )

def neg_rib_mention(report):
    """negatives: any rib-related sentence with fracture-like/abnormal wording that is not a plain negation."""
    ss = [s for s in sentences(report) if re.search(r"\bribs?\b", s, re.I)]
    t = " ".join(ss).lower()
    plain_neg = bool(re.search(r"\bno\b[^.]{0,40}fractur", t))
    return dict(
        n_rib_sent_any=len(ss), neg_rib_text=" | ".join(ss)[:300],
        nr_deform=bool(re.search(r"deformit", t)),
        nr_old_frx=bool(re.search(r"(old|healed|remote|chronic|prior|healing)[^.]{0,40}fractur|fractur[^.]{0,30}(old|healed|remote|chronic)", t)),
        nr_uncert=bool(re.search(r"(possibl|questionable|cannot be excluded|not excluded|equivocal|suspicious|may represent)[^.]{0,60}", t) and "fractur" in t),
        nr_lesion=bool(re.search(r"lesion|lytic|sclerotic|metasta|expansile|enchondroma|bone island", t)),
        nr_frx_word_unneg=bool("fractur" in t and not plain_neg),
    )

def pelv_info(report):
    ss = [s for s in sentences(report) if re.search(r"sacr|pelvi|pubic|ramus|rami|acetab|iliac|ilium|ischi|femor|femur|hip|trochant|coccy", s, re.I) and FRX.search(s)]
    t = " ".join(ss).lower()
    return dict(
        pel_sent=" | ".join(ss)[:400],
        p_sacral_insuff=bool(re.search(r"insufficiency", t) and re.search(r"sacr", t)),
        p_insuff=bool(re.search(r"insufficiency|stress fractur|fragility", t)),
        p_sacrum=bool(re.search(r"sacr", t)), p_coccyx=bool(re.search(r"coccy", t)),
        p_rami=bool(re.search(r"ram(us|i)|pubic|pubis", t)), p_acet=bool(re.search(r"acetab", t)),
        p_iliac=bool(re.search(r"iliac|ilium|ischi", t)),
        p_femur=bool(re.search(r"femor|femur|trochant|hip", t)),
        p_acute=bool(re.search(r"(?<!sub)\bacute\b", t) and "no acute" not in t),
        p_old=bool(re.search(r"\bold\b|healed|remote|chronic|callus|\bprior\b|deformit|sclero|healing|subacute", t)),
        p_hardware=bool(re.search(r"arthroplast|prosthe|hardware|screw|fixation|orif|\brod\b|nail|plate", report or "", re.I)),
        p_uncert=bool(re.search(r"possibl|probabl|may represent|cannot be excluded|not excluded|questionable|suspect|equivocal|indeterminate", t)),
    )
