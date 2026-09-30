# Data access

No patient-level data are included or downloaded by this repository. Numerical method tests and the model example create artificial arrays in memory. Aggregate replay checks use real study summaries, not patient rows. Column names such as `subject_id`, `study_id` and `dicom_id` describe a schema; they are not released records.

Researchers planning a MIMIC application must obtain their own access and follow the applicable PhysioNet requirements and agreements:

- [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/)
- [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)

These are resource pages, not a claim that substituting any available version reconstructs the original study. This release does not include the study's private version manifests or data extraction workflow.

Keep raw tables, radiographs, reports, source labels, linked cohorts, patient identifiers, partition memberships, per-patient hashes, embeddings, scores, predictions, fitted artifacts, credentials and logs outside the public checkout. Reviewed summary statistics and their figures are included; arbitrary derived data are not. The `.gitignore` is a precaution, not proof that a file is safe.

The code-only release does not grant data access or permission to redistribute derivatives. Local study approvals, model-weight terms and access agreements must be assessed for the intended use.
