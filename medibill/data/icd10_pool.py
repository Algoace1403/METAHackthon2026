"""Sample ICD-10-CM diagnosis codes used by MediBill-Env.

All codes are public domain (ICD-10-CM is published by CMS/NCHS, U.S. Government
work, no license required). This module only reproduces a small curated subset
sufficient to generate training data for the 5 SYNTH-PROC-v1 specialties.

Source: CMS 2024 ICD-10-CM tabular files (public domain).
https://www.cms.gov/medicare/icd-10/2024-icd-10-cm
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ICD10Code:
    code: str
    description: str
    specialty: str  # matches a SYNTH-PROC-v1 specialty: CARD / ORTH / GASTRO / RESP / GEN
    prefix: str     # first letter — maps to PolicyRules.covered_icd10_prefixes


_CODES: tuple[ICD10Code, ...] = (
    # ---- CARDIOLOGY (I-prefix) ----
    ICD10Code("I10",    "Essential (primary) hypertension",                         "CARD", "I"),
    ICD10Code("I20.9",  "Angina pectoris, unspecified",                             "CARD", "I"),
    ICD10Code("I21.4",  "Non-ST elevation myocardial infarction (NSTEMI)",          "CARD", "I"),
    ICD10Code("I21.9",  "Acute myocardial infarction, unspecified",                 "CARD", "I"),
    ICD10Code("I25.10", "Atherosclerotic heart disease of native coronary artery",  "CARD", "I"),
    ICD10Code("I48.91", "Unspecified atrial fibrillation",                          "CARD", "I"),
    ICD10Code("I50.9",  "Heart failure, unspecified",                               "CARD", "I"),
    ICD10Code("I44.2",  "Atrioventricular block, complete",                         "CARD", "I"),

    # ---- ORTHOPAEDICS (M and S prefixes) ----
    ICD10Code("M17.11",   "Unilateral primary osteoarthritis, right knee",           "ORTH", "M"),
    ICD10Code("M17.12",   "Unilateral primary osteoarthritis, left knee",            "ORTH", "M"),
    ICD10Code("M23.2",    "Derangement of meniscus due to old tear or injury",       "ORTH", "M"),
    ICD10Code("M51.26",   "Other intervertebral disc displacement, lumbar region",   "ORTH", "M"),
    ICD10Code("M54.5",    "Low back pain",                                           "ORTH", "M"),
    ICD10Code("S52.501A", "Unspecified fracture of lower end of right radius, initial", "ORTH", "S"),
    ICD10Code("S72.001A", "Fracture of unspecified part of neck of right femur, initial", "ORTH", "S"),
    ICD10Code("S83.231A", "Complex tear of medial meniscus of right knee, initial",  "ORTH", "S"),

    # ---- GASTROENTEROLOGY / SURGERY (K-prefix) ----
    ICD10Code("K21.9",  "Gastro-esophageal reflux disease without esophagitis",     "GASTRO", "K"),
    ICD10Code("K25.9",  "Gastric ulcer, unspecified",                               "GASTRO", "K"),
    ICD10Code("K35.80", "Unspecified acute appendicitis",                           "GASTRO", "K"),
    ICD10Code("K40.90", "Unilateral inguinal hernia, without obstruction",          "GASTRO", "K"),
    ICD10Code("K80.20", "Calculus of gallbladder without cholecystitis",            "GASTRO", "K"),
    ICD10Code("K80.10", "Calculus of gallbladder with chronic cholecystitis",       "GASTRO", "K"),
    ICD10Code("K52.9",  "Noninfective gastroenteritis and colitis, unspecified",    "GASTRO", "K"),

    # ---- RESPIRATORY (J-prefix) ----
    ICD10Code("J18.9",   "Pneumonia, unspecified organism",                         "RESP", "J"),
    ICD10Code("J44.9",   "Chronic obstructive pulmonary disease, unspecified",      "RESP", "J"),
    ICD10Code("J45.909", "Unspecified asthma, uncomplicated",                       "RESP", "J"),
    ICD10Code("J96.00",  "Acute respiratory failure, unspecified",                  "RESP", "J"),
    ICD10Code("J84.9",   "Interstitial pulmonary disease, unspecified",             "RESP", "J"),

    # ---- GENERAL MEDICINE (E, N, R prefixes) ----
    ICD10Code("E11.9",  "Type 2 diabetes mellitus without complications",           "GEN", "E"),
    ICD10Code("E11.22", "Type 2 diabetes mellitus with diabetic chronic kidney disease", "GEN", "E"),
    ICD10Code("E78.5",  "Hyperlipidemia, unspecified",                              "GEN", "E"),
    ICD10Code("N18.3",  "Chronic kidney disease, stage 3 (moderate)",               "GEN", "N"),
    ICD10Code("N18.4",  "Chronic kidney disease, stage 4 (severe)",                 "GEN", "N"),
    ICD10Code("R50.9",  "Fever, unspecified",                                       "GEN", "R"),
    ICD10Code("R07.9",  "Chest pain, unspecified",                                  "GEN", "R"),
)


def all_codes() -> list[ICD10Code]:
    return list(_CODES)


def codes_for_specialty(specialty: str) -> list[ICD10Code]:
    target = specialty.upper()
    return [c for c in _CODES if c.specialty == target]


def get_code(code: str) -> ICD10Code:
    for c in _CODES:
        if c.code == code:
            return c
    raise KeyError(f"ICD-10 code '{code}' not in MediBill pool.")
