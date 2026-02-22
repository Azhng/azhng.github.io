SHELL := /bin/zsh

CONTAINER_NAME ?= azhng-jekyll-dev
IMAGE ?= jekyll/jekyll:latest
PORT ?= 4000

JEKYLL_SERVE_ARGS ?= --host 0.0.0.0 --port 4000 --force_polling

.PHONY: start stop logs status restart

start:
	@if docker ps -a --format '{{.Names}}' | grep -Fxq "$(CONTAINER_NAME)"; then \
		docker rm -f "$(CONTAINER_NAME)" >/dev/null; \
	fi
	docker run -d \
		--name "$(CONTAINER_NAME)" \
		-p "$(PORT):4000" \
		-v "$(CURDIR):/srv/jekyll" \
		"$(IMAGE)" \
		jekyll serve $(JEKYLL_SERVE_ARGS)
	@echo "Jekyll server: http://localhost:$(PORT)"

stop:
	@if docker ps -a --format '{{.Names}}' | grep -Fxq "$(CONTAINER_NAME)"; then \
		docker rm -f "$(CONTAINER_NAME)"; \
	else \
		echo "No container named $(CONTAINER_NAME)"; \
	fi

logs:
	docker logs -f "$(CONTAINER_NAME)"

status:
	@docker ps -a --filter "name=$(CONTAINER_NAME)" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

restart: stop start
