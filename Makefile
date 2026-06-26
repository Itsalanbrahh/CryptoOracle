.PHONY: run dev db-init install kronos-setup stop logs

run:
	python3 -m crypto_oracle.main

dev:
	uvicorn crypto_oracle.main:app --reload --host 0.0.0.0 --port 8000

db-init:
	python3 -c "from crypto_oracle.models.db import init_db; import asyncio; asyncio.run(init_db())"

install:
	pip3 install -r requirements.txt

# Clone the Kronos model repo (required for the Kronos agent)
kronos-setup:
	@if [ -d "vendor/kronos" ]; then \
		echo "Kronos repo already at vendor/kronos — pulling latest..."; \
		git -C vendor/kronos pull; \
	else \
		mkdir -p vendor && \
		git clone https://github.com/shiyu-coder/Kronos.git vendor/kronos; \
	fi
	@echo ""
	@echo "Kronos cloned. Add this to your .env:"
	@echo "  KRONOS_REPO_PATH=vendor/kronos"
	@echo ""
	@echo "Models will download from HuggingFace on first oracle run (~100-400MB)."

stop:
	@pkill -f "uvicorn crypto_oracle" || true
	@pkill -f "python -m crypto_oracle" || true
	@echo "Stopped."

logs:
	tail -f crypto_oracle.log
