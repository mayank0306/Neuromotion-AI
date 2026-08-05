# Getting started

1. Copy `.env.example` to `.env` and replace every development secret before deploying.
2. Create a Python 3.11 virtual environment, then install `backend/requirements-dev.txt`.
3. Install the web client with `cd apps/web && npm install`.
4. Start PostgreSQL and Redis with `make docker-up`, then use `make dev`.

Run backend checks with `make test-backend` and web type checks with `make test-frontend`. The repository’s development SQLite configuration is useful for unit tests only; use PostgreSQL for integration testing.
