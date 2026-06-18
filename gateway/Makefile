.PHONY: build test run docker-build clean lint

APP_NAME ?= gateway
DOCKER_IMAGE ?= travelagent/$(APP_NAME)

build:
	go build -o bin/$(APP_NAME) ./cmd/gateway

test:
	go test ./... -v

run:
	go run ./cmd/gateway

docker-build:
	docker build -t $(DOCKER_IMAGE):latest .

clean:
	rm -rf bin/

lint:
	gofmt -w .
	go vet ./...
