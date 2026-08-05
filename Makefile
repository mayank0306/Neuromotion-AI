# NeuroMotion Makefile
# Common development commands

.PHONY: help
help:
	@echo "NeuroMotion Development Commands:"
	@echo "  make install     - Install all dependencies"
	@echo "  make dev         - Start all development servers"
	@echo "  make dev-backend - Start backend server only"
	@echo "  make dev-frontend- Start frontend dev server only"
	@echo "  make dev-ai      - Start AI service"
	@echo "  make test        - Run all tests"
	@echo "  make test-backend - Run backend tests"
	@echo "  make test-frontend- Run frontend tests"
	@echo "  make lint        - Run all linters"
	@echo "  make lint-backend- Lint backend code"
	@echo "  make lint-frontend- Lint frontend code"
	@echo "  make lint-ai     - Lint AI code"
	@echo "  make typecheck   - Run type checking"
	@echo "  make db-init     - Initialize database"
	@echo "  make db-migrate  - Run migrations"
	@echo "  make db-rollback - Rollback migration"
	@echo "  make db-seed     - Seed test data"
	@echo "  make db-reset    - Reset database"
	@echo "  make docker-up   - Start Docker services"
	@echo "  make docker-down - Stop Docker services"
	@echo "  make ai-train    - Train AI models"
	@echo "  make ai-eval     - Evaluate AI models"
	@echo "  make docs        - Generate API docs"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make format      - Format all code"

# Install all dependencies
.PHONY: install
install: install-backend install-frontend install-ai

# Backend dependencies
.PHONY: install-backend
install-backend:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements-dev.txt

# Frontend dependencies
.PHONY: install-frontend
install-frontend:
	@echo "Installing frontend dependencies..."
	cd apps/web && npm install

# AI dependencies
.PHONY: install-ai
install-ai:
	@echo "Installing AI dependencies..."
	pip install -r ai-models/requirements.txt

# Start development servers
.PHONY: dev
dev: docker-up
	@echo "Starting all development servers..."
	cd backend && uvicorn app.main:app --reload --port 8000 &
	$(MAKE) -C apps/web dev &
	@echo "AI inference is provided by the backend pose estimator in development."
	@echo "All servers started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend:  http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

.PHONY: dev-backend
dev-backend:
	cd backend && uvicorn app.main:app --reload --port 8000

.PHONY: dev-frontend
dev-frontend:
	cd apps/web && npm run dev

.PHONY: dev-ai
dev-ai:
	cd ai-models && python -m src.main

# Run tests
.PHONY: test
test: test-backend test-frontend test-integration

.PHONY: test-backend
test-backend:
	cd backend && pytest tests/ -v --cov=app --cov-report=html

.PHONY: test-frontend
test-frontend:
	cd apps/web && npm run test

.PHONY: test-integration
test-integration:
	docker-compose exec backend pytest tests/integration/ -v

.PHONY: test-watch
test-watch:
	cd backend && pytest --watch

# Run linters
.PHONY: lint
lint: lint-backend lint-frontend

.PHONY: lint-backend
lint-backend:
	cd backend && ruff check . && black --check .

.PHONY: lint-frontend
lint-frontend:
	cd apps/web && npm run lint

.PHONY: lint-ai
lint-ai:
	cd ai-models && ruff check . && black --check .

# Type checking
.PHONY: typecheck
typecheck: typecheck-backend typecheck-frontend

.PHONY: typecheck-backend
typecheck-backend:
	cd backend && mypy app/

.PHONY: typecheck-frontend
typecheck-frontend:
	cd apps/web && npx tsc --noEmit

# Database operations
.PHONY: db-init
db-init:
	docker-compose exec db python /scripts/init_db.py

.PHONY: db-migrate
db-migrate:
	docker-compose exec backend alembic upgrade head

.PHONY: db-rollback
db-rollback:
	docker-compose exec backend alembic downgrade -1

.PHONY: db-seed
db-seed:
	docker-compose exec backend python /scripts/seed_data.py

.PHONY: db-reset
db-reset: docker-down
	docker-compose up -d db
	$(MAKE) db-init

# Docker operations
.PHONY: docker-up
docker-up:
	docker-compose up -d

.PHONY: docker-down
docker-down:
	docker-compose down -v

.PHONY: docker-rebuild
docker-rebuild: docker-down
	docker-compose build
	docker-compose up -d

# AI operations
.PHONY: ai-train
ai-train:
	cd ai-models && python train.py --config configs/default.yaml

.PHONY: ai-eval
ai-eval:
	cd ai-models && python evaluate.py --model models/latest

# Documentation
.PHONY: docs
docs:
	@echo "Generating API documentation..."
	@echo "Open http://localhost:8000/docs in your browser"

# Clean
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf backend/.pytest_cache
	rm -rf apps/web/.next
	rm -rf apps/web/node_modules
	rm -rf backend/.venv
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf *.egg-info/

# Format code
.PHONY: format
format:
	cd backend && ruff format . && ruff check --fix .
	cd apps/web && npm run format
