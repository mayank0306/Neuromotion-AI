# NeuroMotion Training Datasets

> **Note:** Datasets are **not** committed to Git. They are shared via the team Drive folder and ignored via `.gitignore`.

## Dataset Overview

| Dataset | Status | Purpose | Source & Citation |
| :--- | :--- | :--- | :--- |
| **REHAB24-6** | **Primary** | Rehab movement quality assessment (65 recordings, 1,072 labeled repetitions, 10 subjects, 6 exercises, binary correct/incorrect labels). | **Source:** https://zenodo.org/records/13305826<br>**License:** Academic or non-profit noncommercial research use only.<br>**Cite:** Černek, Sedmidubsky, Budikova, *"REHAB24-6: Physical Therapy Dataset for Analyzing Pose Estimation Methods"*, SISAP 2024. |
| **KIMORE** | **Planned next** | Real patient data with clinician quality scores (0–100 regression). | Pending integration. |
| **Kaggle Yoga** | **Supporting** | Supporting yoga pose classification module (5 classes). | Kaggle Yoga 5-Class dataset. |

## Expected Folder Structure

```text
datasets/
├── README.md
├── rehab24_6/            # PRIMARY dataset
│   ├── videos/           # Camera17 videos only for now
│   ├── segmentation/     # Segmentation.csv, Segmentation.txt
│   └── landmarks/        # Empty; MediaPipe output will be cached here later
├── kimore/               # PLANNED (empty placeholder with .gitkeep)
└── kaggle_yoga/
    └── images/           # Supporting yoga module images
```
