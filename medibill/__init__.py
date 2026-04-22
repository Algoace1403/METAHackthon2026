"""MediBill-Env — OpenEnv environment for cashless health-insurance claim reconciliation.

Round 2 submission for the Meta x Scaler OpenEnv Hackathon. Simulates the IRDAI
1-hour pre-auth / 3-hour discharge clock with mid-episode policy drift.

Public API re-exports:
    MediBillAction, MediBillObservation, MediBillState  (models)
    MediBillEnvironment                                  (server/environment)
    MediBillEnv                                          (client)

All medical coding is either public-domain (ICD-10-CM, LOINC, RxNorm, HCPCS
Level II, CGHS rates) or synthetic (SYNTH-PROC-v1). No AMA CPT content is
used anywhere in this package.
"""

__version__ = "0.1.0"
