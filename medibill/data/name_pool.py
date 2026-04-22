"""Fully synthetic name pools for MediBill-Env patient/hospital/physician records.

All names are generic or common Indian names chosen to provide regional
diversity. No individual is referenced. Do not map any entry in this file to
a real person, institution, or identifier.
"""

from __future__ import annotations


# Patient first names — broad Indian + international mix for duplicate detection tests
FIRST_NAMES_MALE: tuple[str, ...] = (
    "Rahul", "Arjun", "Vikram", "Rohit", "Amit", "Sanjay", "Karan", "Ravi",
    "Suresh", "Nitin", "Deepak", "Vivek", "Prakash", "Ajay", "Manish",
    "Ashok", "Rajesh", "Anil", "Mohan", "Aditya",
    "Abdul", "Ibrahim", "Faisal", "Zaid",
    "Aaron", "Daniel", "Michael",
)

FIRST_NAMES_FEMALE: tuple[str, ...] = (
    "Priya", "Sneha", "Anjali", "Kavita", "Meera", "Pooja", "Neha", "Divya",
    "Shalini", "Aarti", "Rekha", "Sunita", "Lakshmi", "Nisha", "Ritu",
    "Kiran", "Rashmi", "Deepika", "Swati", "Anita",
    "Fatima", "Aisha", "Zainab",
    "Sarah", "Emily", "Grace",
)

# Patient last names — regional India mix
LAST_NAMES: tuple[str, ...] = (
    "Sharma", "Verma", "Gupta", "Patel", "Iyer", "Nair", "Reddy", "Rao",
    "Khan", "Ahmed", "Sheikh", "Sayed",
    "Kumar", "Singh", "Mehta", "Agarwal", "Bansal", "Chawla", "Kapoor",
    "Mishra", "Tripathi", "Shukla",
    "Das", "Bose", "Chatterjee", "Banerjee", "Mukherjee",
    "Menon", "Pillai", "Naik",
    "D'Souza", "Fernandes", "Pinto",
)

# 15 fictional hospitals spread across major metros; names are composed of
# generic descriptive terms. None corresponds to a specific real institution.
HOSPITALS: tuple[dict[str, str], ...] = (
    {"hospital_id": "HOSP-MUM-001", "name": "Coastal Multispeciality Hospital",   "city": "Mumbai"},
    {"hospital_id": "HOSP-MUM-002", "name": "Andheri Care Center",                "city": "Mumbai"},
    {"hospital_id": "HOSP-DEL-001", "name": "Capital District Hospital",          "city": "Delhi"},
    {"hospital_id": "HOSP-DEL-002", "name": "Yamuna Super-Speciality Trust",      "city": "Delhi"},
    {"hospital_id": "HOSP-BLR-001", "name": "Cauvery Medical Institute",          "city": "Bengaluru"},
    {"hospital_id": "HOSP-BLR-002", "name": "Whitefield General Hospital",        "city": "Bengaluru"},
    {"hospital_id": "HOSP-CHN-001", "name": "Marina Cardiac Hospital",            "city": "Chennai"},
    {"hospital_id": "HOSP-CHN-002", "name": "Velachery Multispeciality",          "city": "Chennai"},
    {"hospital_id": "HOSP-HYD-001", "name": "Deccan Heart Institute",             "city": "Hyderabad"},
    {"hospital_id": "HOSP-HYD-002", "name": "Jubilee Medical Center",             "city": "Hyderabad"},
    {"hospital_id": "HOSP-KOL-001", "name": "Ganges Multispeciality",             "city": "Kolkata"},
    {"hospital_id": "HOSP-PUN-001", "name": "Deccan Orthopaedic Hospital",        "city": "Pune"},
    {"hospital_id": "HOSP-AHM-001", "name": "Sabarmati Care Center",              "city": "Ahmedabad"},
    {"hospital_id": "HOSP-JAI-001", "name": "Aravalli District Hospital",         "city": "Jaipur"},
    {"hospital_id": "HOSP-LUC-001", "name": "Gomti General Hospital",             "city": "Lucknow"},
)


def all_first_names(gender: str) -> tuple[str, ...]:
    g = gender.upper()
    if g in ("M", "MALE"):
        return FIRST_NAMES_MALE
    if g in ("F", "FEMALE"):
        return FIRST_NAMES_FEMALE
    # "O" / unspecified → combined pool
    return FIRST_NAMES_MALE + FIRST_NAMES_FEMALE


def all_last_names() -> tuple[str, ...]:
    return LAST_NAMES


def all_hospitals() -> tuple[dict[str, str], ...]:
    return HOSPITALS
