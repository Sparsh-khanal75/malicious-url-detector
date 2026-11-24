.PHONY: install lint test train train-quick serve docker-build docker-run up down

install:        ## editable install with dev tools
	pip install -e ".[dev]"
lint:
	ruff check . && ruff format --check .
test:
	pytest -q
train:          ## full dataset, ~6 min on 2 cores
	python -m urlguard.train
train-quick:    ## 40k-row sample + quality gate, ~30 s
	python -m urlguard.train --sample 40000 --min-pr-auc 0.90
serve:
	uvicorn urlguard.api:app --reload --port 8000
docker-build:
	docker build -t urlguard:latest .
docker-run:
	docker run --rm -p 8000:8000 urlguard:latest
up:             ## API + Prometheus
	docker compose up --build -d
down:
	docker compose down
