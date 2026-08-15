# Deploying a Machine Learning Model (Python + FastAPI)

An advanced project in Python that takes a trained scikit-learn model from a pickle on disk to a production prediction service. You train and persist a small model, build a load-and-predict module that loads the artifact once, wrap it in a FastAPI endpoint with strict Pydantic input validation, support both single and batch inference, add a model registry so versions are explicit and switchable, containerize the service with a Dockerfile, instrument it with latency and basic drift monitoring, write a health/readiness probe and a load test, and finish with a docker-compose stack and an AWS deployment runbook. The focus is operational: not how the model learns, but how you ship, serve, version, watch, and scale it.

Built step-by-step with [KhwajaLabs Build](https://khwajalabs.com).

## Stack
- Python
- scikit-learn
- FastAPI
- Pydantic
- Docker
- Prometheus
- AWS
