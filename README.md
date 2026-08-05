# NeuroMotion

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-blue?logo=typescript)](https://www.typescriptlang.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-0096D9?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB?logo=react)](https://reactjs.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue?logo=postgresql)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-24.0%2B-2496ED?logo=docker)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-85%2B%20coverage-brightgreen)](tests/)
[![CI](https://github.com/username/neuromation/actions/workflows/ci.yml/badge.svg)](https://github.com/username/neuromation/actions/workflows/ci.yml)
[![Security](https://img.shields.io/badge/security-SAST%20%7C%20DAST-green)](SECURITY.md)

An AI-powered digital healthcare platform focused on human movement intelligence. Provides AI-powered yoga analysis, posture correction, gait analysis, movement tracking, and personalized health insights.

## Features

- **AI-Powered Yoga Analysis** - Real-time pose detection and form correction
- **Posture Assessment** - Static posture analysis with personalized recommendations
- **Gait Analysis** - Walking pattern analysis for injury prevention
- **Progress Tracking** - Comprehensive dashboard with health metrics
- **Personalized Insights** - AI-generated health recommendations
- **Telemedicine** - Video consultations with healthcare professionals (planned)

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/username/neuromotion.git
cd neuromotion

# 2. Install dependencies
make install

# 3. Set up environment
cp .env.example .env.local
# Edit .env.local with your values

# 4. Start local infrastructure
make docker-up

# 5. Initialize database
make db-init

# 6. Start development servers
make dev
```

Then visit:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Adminer (DB): http://localhost:8080

## Documentation

- [Architecture Documentation](docs/architecture/)
- [API Documentation](docs/api/)
- [Development Guide](docs/development/)
- [User Guide](docs/user-guide/)

## Project Structure

```
neuromotion/
├── backend/                 # FastAPI backend
├── apps/web/                # Next.js frontend
├── ai-models/               # AI model definitions
├── infrastructure/          # Docker, Kubernetes, Terraform
├── docs/                    # Documentation
├── tests/                   # End-to-end tests
├── scripts/                 # Utility scripts
├── docker-compose.yml       # Local development setup
├── Makefile                 # Development commands
└── .env.example             # Environment template
```

## Contributing

See our [Contributing Guide](docs/development/contributing.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for pose estimation
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [Next.js](https://nextjs.org/) for the frontend framework
- [PostgreSQL](https://www.postgresql.org/) for the database
- [Render](https://render.com/) for free hosting
