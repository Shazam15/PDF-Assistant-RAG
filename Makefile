.PHONY: dev-backend dev-frontend dev-worker dev dev-wsl dev-ubuntu doctor-wsl doctor-ubuntu test migrate lint format install install-backend install-backend-wsl install-backend-ubuntu install-frontend build clean docker-up docker-down docker-logs help

BACKEND_DIR = backend
FRONTEND_DIR = frontend
BACKEND_PORT ?= 7860
PYTHON ?= python3

help:
	@echo "Usage:"
	@echo "  make dev-backend     Start FastAPI (uvicorn) on port $(BACKEND_PORT)"
	@echo "  make dev-frontend    Start Next.js dev server on port 3000"
	@echo "  make dev-worker      Start one Celery document worker"
	@echo "  make dev             Start both backend and frontend concurrently"
	@echo "  make dev-wsl         Validate Windows Ollama/PostgreSQL and start in WSL2"
	@echo "  make dev-ubuntu      Validate local T4/Ollama/PostgreSQL/Redis and start on Ubuntu"
	@echo "  make doctor-wsl      Validate the WSL2, Windows Ollama, and PostgreSQL split"
	@echo "  make doctor-ubuntu   Validate the native Ubuntu/T4 runtime"
	@echo "  make test            Run pytest"
	@echo "  make migrate         Apply database migrations"
	@echo "  make lint            Run flake8 (backend) + eslint (frontend)"
	@echo "  make format          Auto-format Python with black (backend)"
	@echo "  make install         Install all dependencies (backend + frontend)"
	@echo "  make install-backend Install Python dependencies"
	@echo "  make install-backend-wsl Install CPU PyTorch and backend dependencies in WSL2"
	@echo "  make install-backend-ubuntu Install CPU PyTorch and backend dependencies on Ubuntu"
	@echo "  make install-frontend Install Node.js dependencies"
	@echo "  make build           Build frontend for production"
	@echo "  make clean           Remove __pycache__, .next, build artifacts"
	@echo "  make docker-up       Start PostgreSQL/pgvector and Redis"
	@echo "  make docker-down     Stop all Docker services"
	@echo "  make docker-logs     Tail Docker logs"

dev-backend:
	cd $(BACKEND_DIR) && $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port $(BACKEND_PORT) --reload

dev-frontend:
	cd $(FRONTEND_DIR) && npm run dev

dev-worker:
	cd $(BACKEND_DIR) && $(PYTHON) -m celery -A app.celery_app.celery_app worker --loglevel=info --concurrency=1

dev:
	@echo "Starting backend (port $(BACKEND_PORT)) and frontend (port 3000)..."
	npx concurrently --kill-others --names "BACKEND,FRONTEND" --prefix-colors "blue,green" \
		"$(MAKE) dev-backend" \
		"$(MAKE) dev-frontend"

doctor-wsl:
	@WINDOWS_HOST="$$(ip route show default 2>/dev/null | awk '/default/ {print $$3; exit}')"; \
	if [ -z "$$WINDOWS_HOST" ] && [ -z "$$OLLAMA_BASE_URL" ]; then \
		echo "[FAIL] Could not resolve the Windows host from the WSL default route."; exit 1; \
	fi; \
	OLLAMA_URL="$${OLLAMA_BASE_URL:-http://$$WINDOWS_HOST:11434}"; \
	echo "Using Windows Ollama at $$OLLAMA_URL"; \
	cd $(BACKEND_DIR) && MODEL_PROFILE=wsl_t4 OLLAMA_BASE_URL="$$OLLAMA_URL" \
	$(PYTHON) -m app.runtime_doctor --profile wsl_t4

dev-wsl:
	@WINDOWS_HOST="$$(ip route show default 2>/dev/null | awk '/default/ {print $$3; exit}')"; \
	if [ -z "$$WINDOWS_HOST" ] && [ -z "$$OLLAMA_BASE_URL" ]; then \
		echo "[FAIL] Could not resolve the Windows host from the WSL default route."; exit 1; \
	fi; \
	OLLAMA_URL="$${OLLAMA_BASE_URL:-http://$$WINDOWS_HOST:11434}"; \
	MODEL_PROFILE=wsl_t4 OLLAMA_BASE_URL="$$OLLAMA_URL" \
	$(MAKE) doctor-wsl PYTHON="$(PYTHON)" && \
	MODEL_PROFILE=wsl_t4 OLLAMA_BASE_URL="$$OLLAMA_URL" \
	$(MAKE) dev PYTHON="$(PYTHON)"

doctor-ubuntu:
	@cd $(BACKEND_DIR) && \
	MODEL_PROFILE=ubuntu_t4 \
	OLLAMA_BASE_URL="$${OLLAMA_BASE_URL:-http://127.0.0.1:11434}" \
	$(PYTHON) -m app.runtime_doctor --profile ubuntu_t4

dev-ubuntu:
	@$(MAKE) doctor-ubuntu PYTHON="$(PYTHON)"
	@echo "Starting Ubuntu/T4 backend, frontend, and document worker..."
	npx concurrently --kill-others --names "BACKEND,FRONTEND,WORKER" --prefix-colors "blue,green,yellow" \
		"$(MAKE) dev-backend PYTHON='$(PYTHON)'" \
		"$(MAKE) dev-frontend" \
		"$(MAKE) dev-worker PYTHON='$(PYTHON)'"

test:
	cd $(BACKEND_DIR) && $(PYTHON) -m pytest -v

migrate:
	cd $(BACKEND_DIR) && $(PYTHON) -c "from app.database import init_db; init_db()" && $(PYTHON) -m alembic upgrade head

lint:
	cd $(BACKEND_DIR) && $(PYTHON) -m flake8 .
	cd $(FRONTEND_DIR) && npm run lint

format:
	cd $(BACKEND_DIR) && $(PYTHON) -m black .

install: install-backend install-frontend

install-backend:
	$(PYTHON) -m pip install -r $(BACKEND_DIR)/requirements.txt

install-backend-wsl:
	# torch and torchvision must come from the same index in the same
	# command so pip resolves a matched pair — docling (in requirements.txt)
	# pulls torchvision transitively, and installing it separately from
	# default PyPI produces a torch/torchvision ABI mismatch
	# ("operator torchvision::nms does not exist").
	$(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
	$(PYTHON) -m pip install -r $(BACKEND_DIR)/requirements.txt

install-backend-ubuntu:
	$(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
	$(PYTHON) -m pip install -r $(BACKEND_DIR)/requirements.txt

install-frontend:
	cd $(FRONTEND_DIR) && npm install

build:
	cd $(FRONTEND_DIR) && npm run build

clean:
	rm -rf $(BACKEND_DIR)/__pycache__
	find $(BACKEND_DIR) -type d -name __pycache__ -exec rm -rf {} +
	rm -rf $(FRONTEND_DIR)/.next
	rm -rf $(FRONTEND_DIR)/out
	rm -rf $(FRONTEND_DIR)/build
	rm -rf .pytest_cache

docker-up:
	docker compose up -d postgres redis

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
