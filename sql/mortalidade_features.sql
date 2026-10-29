-- Cohort and features used in the experiments (MIMIC-IV v3.1, PostgreSQL).
-- Definition recovered verbatim (pg_get_viewdef) from the authors' local view
-- mimiciv_hosp.mortalidade_features; the logic is unchanged.
--
-- Unit of analysis: hospital admission (hadm_id). All admissions of patients with
-- anchor_age >= 18 are included; there is NO restriction to ICU stays or to first admissions.
-- Expected result: 546,028 rows, 2.16% with hospital_expire_flag = 1.
--
-- Notes:
--   * age = patients.anchor_age (the patient's anchor age, not age at admission);
--   * first_careunit = first unit in transfers (includes emergency department and wards);
--   * num_diagnoses / num_procedures count ICD codes over the whole admission and are
--     only complete at discharge.

CREATE VIEW mimiciv_hosp.mortalidade_features AS
WITH first_careunit AS (
    SELECT t.hadm_id,
           t.careunit AS first_careunit
    FROM (
        SELECT transfers.hadm_id,
               transfers.careunit,
               row_number() OVER (PARTITION BY transfers.hadm_id ORDER BY transfers.intime) AS rn
        FROM mimiciv_hosp.transfers
    ) t
    WHERE t.rn = 1
), coorte AS (
    SELECT p.subject_id,
           a.hadm_id,
           p.anchor_age AS age,
           p.gender,
           a.race,
           a.insurance,
           a.hospital_expire_flag,
           fcu.first_careunit
    FROM mimiciv_hosp.admissions a
    JOIN mimiciv_hosp.patients p ON a.subject_id = p.subject_id
    LEFT JOIN first_careunit fcu ON a.hadm_id = fcu.hadm_id
    WHERE p.anchor_age >= 18
), diagnoses_agg AS (
    SELECT diagnoses_icd.hadm_id,
           count(diagnoses_icd.icd_code) AS num_diagnoses
    FROM mimiciv_hosp.diagnoses_icd
    GROUP BY diagnoses_icd.hadm_id
), procedures_agg AS (
    SELECT procedures_icd.hadm_id,
           count(procedures_icd.icd_code) AS num_procedures
    FROM mimiciv_hosp.procedures_icd
    GROUP BY procedures_icd.hadm_id
)
SELECT c.hadm_id,
       c.age,
       c.gender,
       c.race,
       c.insurance,
       c.first_careunit,
       COALESCE(da.num_diagnoses, 0::bigint) AS num_diagnoses,
       COALESCE(pa.num_procedures, 0::bigint) AS num_procedures,
       c.hospital_expire_flag
FROM coorte c
LEFT JOIN diagnoses_agg da ON c.hadm_id = da.hadm_id
LEFT JOIN procedures_agg pa ON c.hadm_id = pa.hadm_id;

-- Export (psql) of the two files expected in data/:
-- \copy (SELECT * FROM mimiciv_hosp.mortalidade_features) TO 'data/mortalidade_features.csv' CSV HEADER
-- \copy (SELECT hadm_id, subject_id FROM mimiciv_hosp.admissions ORDER BY hadm_id) TO 'data/hadm_subject_map.csv' CSV HEADER
