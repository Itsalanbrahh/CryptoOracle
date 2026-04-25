.PHONY: run dev db-init stop logs install

run:
	python -m crypto_oracle.main

dev:
	uvicorn crypto_oracle.main:app --reload --host 0.0.0.0 --port 8000

db-init:
	python -c "from crypto_oracle.models.db import init_db; import asyncio; asyncio.run(init_db())"

install:
	pip install -r requirements.txt

stop:
	@pkill -f "uvicorn crypto_oracle" || true
	@pkill -f "python -m crypto_oracle" || true
	@echo "Stopped."

logs:
	tail -f crypto_oracle.log
