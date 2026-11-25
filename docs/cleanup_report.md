# NeuroMotion Codebase Cleanup Report

This report documents the audit of the NeuroMotion codebase against the current plan:
**video upload $\to$ OpenCV $\to$ MediaPipe BlazePose $\to$ shared feature engineering $\to$ two trained PyTorch MLPs (rehab model, yoga classifier) + rule-based posture/gait/balance/flexibility modules $\to$ explainable output.**

Checkpoint commit created prior to cleanup: `ecf3299`

---

## 1. Planned Removals

### Batch 1: Stray Duplicates, Junk Files & Empty Out-of-Scope Directories
- **`test_ai_service.py` (root)**: Duplicate of `backend/app/tests/test_ai_service.py` (which is newer and resides inside `backend/`).
- **`backend/app/analysis/test_feature_engineering.py`**: Duplicate of `backend/tests/unit/test_feature_engineering.py`. Keeping the one in `backend/tests/unit/`.
- **`codex`**: Stray 56-line handoff note from an earlier session (`2026-08-05`) at the repository root.
- **Empty out-of-scope AI model directories** (no files inside; gait/posture are deterministic geometry, BlazePose is pretrained):
  - `ai-models/gait-analysis/`
  - `ai-models/posture-correction/`
  - `ai-models/pose-estimation/` (`evaluation/`, `inference/`, `models/`, `training/`)
  - `ai-models/ml-pipelines/` (active pipelines are in `backend/app/ai-models/ml-pipelines/`)
  - `backend/app/ai-models/pose-estimation/`
- **Empty placeholder directory trees** (contain zero files):
  - `services/` (`notification-service`, `payment-service`, `telemedicine-service`, `analytics-service`, `rehab-service`, `movement-analysis-service`, `user-service`, `gateway`)
  - `packages/` (`middleware`, `database`, `crypto`, `logger`, `config`, `constants`, `healthcheck`, `validators`)
  - `apps/admin/`, `apps/mobile/`
  - `tests/` (root empty `unit/`, `integration/`; real tests are in `backend/tests/`)
  - `data/` (`migrations/`, `schemas/`, `scripts/`)
  - `scripts/`, `backend/app/domain/`, `backend/docs/`, `backend/scripts/`
- **Generated / junk files**:
  - `.DS_Store`, `infrastructure/terraform/.DS_Store`
  - `.coverage`, `backend/.coverage`, `backend/htmlcov/`, `backend/.pytest_cache/`, `__pycache__/`

### Batch 2: Dead Code & Unused Config Entries (Verified with `grep`)
- **`backend/app/ai/pose_estimator.py` & `backend/app/ai/__init__.py`**: `PoseEstimator` and `get_pose_estimator` are no longer referenced anywhere in `backend/app/` (`backend/app/api/deps.py` uses `AIService`).
  - Also remove unused `mock_pose_estimator` fixture from `backend/tests/conftest.py`.
- **`backend/app/utils/export.py` (`DataExporter`)**: Unused dead code; never imported or called anywhere outside `backend/app/utils/__init__.py`.
- **`backend/app/utils/image.py` (`ImageProcessor`)**: Unused dead code; never imported or called anywhere outside `backend/app/utils/__init__.py`.
- **`backend/app/utils/video.py` (`VideoProcessor`)**: Unused dead code; `AIService.process_video_frames` uses `cv2.VideoCapture` directly.
  - Clean up `backend/app/utils/__init__.py` accordingly (keeping `validators.py` which is tested by `backend/tests/unit/test_validators.py`).
- **Unused `backend/app/core/config.py` entries** (verified zero references in codebase):
  - `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM_EMAIL`
  - `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_SUCCESS_URL`, `STRIPE_CANCEL_URL`
  - `SENTRY_DSN`, `PROMETHEUS_ENABLED`, `ENABLE_TRACING`
  - `ENABLE_GAIT_ANALYSIS`, `ENABLE_POSTURE_ASSESSMENT`, `ENABLE_DATA_EXPORT`, `ENABLE_DEBUG_TOOLBAR`
- **`backend/app/main.py` OpenAPI description**: Update stale description ("AI-powered yoga analysis") to reflect the rehab movement quality assessment platform.

### Batch 3: Unused Dependencies & `.gitignore`
- **`backend/requirements.txt`**:
  - Remove unused packages: `aioredis` (redundant with `redis`), `aiohttp`, `ffmpeg-python`, `PyPDF2`, `python-docx`, `openpyxl`, `sentry-sdk`, `celery`, `flower`, `python-slugify`, duplicate `python-jose[cryptography]`, and duplicate `python-multipart`.
  - Add missing runtime dependency: `torch>=2.0.0`.
- **`.gitignore`**:
  - Ensure `backend/.pytest_cache/`, `**/.DS_Store`, `htmlcov/`, `.coverage` are ignored.

---

## 2. Unsure List (Left Untouched for Review)

The following items are **not** deleted because they may be intentionally retained for frontend integration, future phases, or reference:

1. **`IMPLEMENTATION_SUMMARY.md` and `PHASE1_COMPLETION_SUMMARY.md` (root)**: Historical Phase 1 summary documents left by the previous tool. May be useful as history or moved to `docs/` if desired.
2. **`apps/web/`**: Next.js 14 frontend application. Kept intact as the web client.
3. **`infrastructure/`**: Terraform, Kubernetes, and Docker configuration starters. Kept intact for deployment.
4. **`backend/app/api/v1/endpoints/insights.py`**: Returns placeholder dashboard/insights/trends data, but is wired into `backend/app/api/v1/__init__.py` and may be used by `apps/web/`.
5. **`backend/app/middleware/auth.py` & `backend/app/middleware/logging.py`**: Currently not registered in `main.py`, but kept in case middleware registration is planned.
6. **`backend/app/core/config.py` (`ENABLE_TELEMEDICINE`)**: Referenced in `backend/app/core/database.py` line 165 (`redis_client = RedisClient() if not settings.ENABLE_TELEMEDICINE else None`). Kept to avoid altering database initialization.
7. **`backend/app/data/exercises.py`**: Exercise catalog data; not currently imported by `AIService`, but useful reference metadata for rehab & yoga exercises.
8. **`backend/migrations/` & `backend/app/workers/`**: Empty Alembic/worker scaffolds reserved for database migrations and background jobs.
9. **`ai-models/configs/default.yaml` & `ai-models/README.md`**: Baseline config and docs for `ai-models/`.
10. **`.claude/`**: Local Claude Code settings/hooks directory.
