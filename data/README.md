# data/

This folder is intentionally empty in the repository. MIMIC-IV is distributed by PhysioNet under the Credentialed Health Data Use Agreement. That agreement forbids redistributing the data or patient-level files derived from it, so no CSV, pickle or model trained on it is versioned here (see `.gitignore`).

To reproduce the experiments:

1. Obtain credentialed access to MIMIC-IV v3.1 at https://physionet.org/content/mimiciv/.
2. Load MIMIC-IV into PostgreSQL. The official scripts are at https://github.com/MIT-LCP/mimic-code.
3. Create the view and export the two files using [`../sql/mortalidade_features.sql`](../sql/mortalidade_features.sql):

| File | Rows | Contents |
|---|---|---|
| `mortalidade_features.csv` | 546,028 | hadm_id, age, gender, race, insurance, first_careunit, num_diagnoses, num_procedures, hospital_expire_flag |
| `hadm_subject_map.csv` | 546,028 | hadm_id → subject_id (used only by the patient-level split in `run_baselines.py`) |

The experiment scripts write the partitioned federations to `data/cache/`. These are patient-level derived data and are likewise not versioned.
