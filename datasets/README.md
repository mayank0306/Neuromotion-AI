# Training Datasets

Place your local datasets here (ignored by Git via `.gitignore`):

```text
datasets/
├── ui-prmd/          # UI-PRMD rehabilitation dataset (classification: correct/incorrect)
├── kimore/           # KIMORE rehabilitation dataset (regression: clinician quality score)
└── kaggle_yoga/      # Kaggle Yoga 5-Class dataset
    ├── images/       # Pose images
    └── labels.csv    # Image filenames and class labels
```

Set `DATASET_PATH=./datasets` in your `.env` (or export `DATASET_PATH=/Users/mayank/projects/neuromotion/datasets`) so the training configs resolve these paths automatically.
