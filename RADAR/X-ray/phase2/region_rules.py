"""
Rule-based mapping from PadChest's `Localizations` vocabulary (104 distinct
"loc X" strings, enumerated by scanning the actual label CSV -- see
RADAR/X-ray/phase2/README.md for the frequency table) to RADAR-X-ray's 3
anatomical regions: left_lung, right_lung, heart (heart doubles as
"heart/mediastinum" per RADAR/X-ray/PLAN.md, since we don't have a separate
mediastinum mask).

Anything not listed here (ribs, spine, shoulder girdle, abdomen, soft
tissue, etc.) intentionally maps to no region -- those findings still count
toward the whole-image caption, just not toward any per-region caption.
"""

LEFT_LUNG_LOCS = {
    "loc left",
    "loc left lower lobe",
    "loc left upper lobe",
    "loc left costophrenic angle",
    "loc lingula",  # anatomically left-lung-only structure
}

RIGHT_LUNG_LOCS = {
    "loc right",
    "loc right upper lobe",
    "loc right lower lobe",
    "loc right costophrenic angle",
    "loc middle lobe",  # the middle lobe only exists in the right lung
    "loc minor fissure",  # the minor/horizontal fissure is right-lung-only
}

# lung-related but no laterality given in the token itself -> both lungs
BILATERAL_LUNG_LOCS = {
    "loc bilateral",
    "loc basal",
    "loc basal bilateral",
    "loc apical",
    "loc hilar",
    "loc hilar bilateral",
    "loc perihilar",
    "loc costophrenic angle",
    "loc middle lung field",
    "loc lower lobe",
    "loc upper lobe",
    "loc lower lung field",
    "loc upper lung field",
    "loc lung field",
    "loc lobar",
    "loc subsegmental",
    "loc fissure",
    "loc major fissure",  # present in both lungs (unlike the minor fissure)
    "loc bronchi",
    "loc central",
    "loc infrahilar",
    "loc suprahilar",
    "loc subpleural",
    "loc peribronchi",
    "loc airways",
    "loc diffuse bilateral",
    "loc bilateral costophrenic angle",
    "loc supradiaphragm",
}

# heart doubles as "heart/mediastinum" -- mediastinal vessels/structures
# bucket here since we don't have a standalone mediastinum mask
HEART_LOCS = {
    "loc cardiac",
    "loc retrocardiac",
    "loc paracardiac",
    "loc cardiophrenic angle",
    "loc mediastinum",
    "loc superior mediastinum",
    "loc lower mediastinum",
    "loc middle mediastinum",
    "loc posterior mediastinum",
    "loc anterior mediastinum",
    "loc paramediastinum",
    "loc aortic",
    "loc supra aortic",
    "loc aortic button",
    "loc aortopulmonary window",
    "loc superior cave vein",
    "loc brachiocephalic veins",
    "loc subclavian vein",
    "loc pulmonary artery",
    "loc coronary",
    "loc tracheal",
    "loc paratracheal",
    "loc thymus",
    "loc esophageal",
}

REGION_LOC_MAP = {
    "left_lung": LEFT_LUNG_LOCS,
    "right_lung": RIGHT_LUNG_LOCS,
    "heart": HEART_LOCS,
}
BILATERAL_REGIONS = ("left_lung", "right_lung")

# findings that are heart/mediastinum-specific by NAME, often reported as a
# standalone Label with no "loc cardiac"/"loc mediastinum" tag at all
# (e.g. "cardiomegaly" appears 8,478 times but "loc cardiac" only 13,067 --
# many cardiomegaly rows carry no location tag whatsoever)
HEART_LABEL_KEYWORDS = (
    "cardi",       # cardiomegaly, cardiophrenic, cardiac...
    "aort",        # aortic elongation, aortic atheromatosis, aortic button...
    "mediastin",   # mediastinal enlargement/mass, mediastinic lipomatosis
    "pacemaker",
    "sternotomy",  # post-sternotomy changes are heart-surgery-related
    "pericard",
    "heart",       # heart insufficiency, artificial heart valve...
)


def locs_to_regions(loc_tokens):
    """loc_tokens: iterable of normalized (stripped, lowercase) 'loc X' strings.
    Returns the set of region names ('left_lung'/'right_lung'/'heart') implied."""
    regions = set()
    for loc in loc_tokens:
        for region, loc_set in REGION_LOC_MAP.items():
            if loc in loc_set:
                regions.add(region)
        if loc in BILATERAL_LUNG_LOCS:
            regions.update(BILATERAL_REGIONS)
    return regions


def label_implies_heart(label_text):
    text = label_text.lower()
    return any(kw in text for kw in HEART_LABEL_KEYWORDS)
