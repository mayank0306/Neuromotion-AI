# NeuroMotion Development Guide

## Project Overview

NeuroMotion is an AI-powered digital healthcare platform for human movement intelligence.
This document contains essential context for all AI coding assistants working on this project.

## Architecture Summary

The project follows **Clean Architecture** principles with clear separation of concerns:

```
frontend/     → REST/WebSocket   → backend/        → AI Models (Python)
(Next.js)      API Gateway         (FastAPI)        (MediaPipe/PyTorch)
```

### Key Decisions
- **Monolith with Modular Design** - Single deployable unit with clean boundaries
- **Clean Architecture** - Domain-driven, testable, framework-independent
- **Free Technologies Only** - PostgreSQL, Redis, MediaPipe, FastAPI, Next.js
- **Healthcare-Focused** - HIPAA-aware design, field-level encryption for PHI

## Directory Structure

```
neuromotion/
├── backend/              # FastAPI application (main backend)
├── frontend/             # Next.js 14 application
├── ai-models/            # AI/ML model definitions and pipelines
├── infrastructure/       # Docker, Kubernetes, Terraform configs
├── docs/                 # Project documentation
├── tests/                # End-to-end tests
└── scripts/              # Utility scripts
```

## Development Workflow

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Write tests first (TDD approach)
3. Implement the feature
4. Run tests: `make test`
5. Run linting: `make lint`
6. Run type checking: `make typecheck`
7. Commit with descriptive message: `git commit -m "feat: add my feature"`
8. Push and create pull request

## Coding Standards

### Python Backend (FastAPI)
- PEP 8, PEP 257, PEP 484 compliant
- Use type hints everywhere
- Docstrings for all public functions
- Pydantic models for validation
- Repository pattern for data access
- Dependency injection with FastAPI Depends

### TypeScript Frontend
- TypeScript strict mode enabled
- Path aliases for imports (@/* maps to src/*)
- Consistent naming: camelCase for variables, PascalCase for components
- Functional components with hooks
- Custom hooks for logic reuse

### AI Models
- PyTorch for deep learning
- Type hints throughout
- Experiment tracking with MLflow
- Model versioning

## Testing Strategy

```
Tests are organized in:
├── backend/tests/
│   ├── unit/              # Individual function/class tests
│   ├── integration/       # Component integration tests
│   └── conftest.py        # Test configuration
├── frontend/tests/
│   ├── unit/              # Component unit tests
│   ├── integration/       # Feature integration tests
│   └── e2e/               # End-to-end browser tests
└── tests/                  # Cross-system e2e tests
```

Target coverage: 85% minimum for unit tests

## Common Commands

| Command | Description |
|---------|-------------|
| `make dev` | Start all development servers |
| `make test` | Run all tests |
| `make lint` | Run all linters |
| `make typecheck` | Run type checking |
| `make db-init` | Initialize database |
| `make db-migrate` | Run database migrations |
| `make docker-up` | Start Docker services |
| `make docker-down` | Stop Docker services |
| `make ai-train` | Train AI models |
| `make ai-eval` | Evaluate AI models |

## API Conventions

- RESTful API design with JSON
- Versioning: `/api/v1/` prefix
- Authentication: JWT Bearer tokens
- Error handling: RFC 7807 Problem Details
- Pagination: Cursor-based for lists
- Rate limiting: 100 requests/minute per user

## Environment Variables

Key environment variables (see `.env.example` for all):

```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/neuromotion
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=your-secret-key
STRIPE_SECRET_KEY=sk_test_...
```

## Code Style Rules

1. Use descriptive names - avoid abbreviations
2. One responsibility per function/class
3. Fail fast - validate inputs early
4. Log important operations with context
5. Handle all error cases explicitly
6. Write tests for all new features
7. Document complex algorithms

## AI Model Guidelines

1. Save models with version numbers
2. Document model architecture and hyperparameters
3. Track experiments with MLflow
4. Validate with diverse datasets
5. Check for bias across demographics

## Deployment Notes

- Production builds use Docker
- Kubernetes manifests in infrastructure/
- Free tier deployments on Render/Vercel
- Environment-specific configs in overlays/
