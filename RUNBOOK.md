# Deployment Runbook — Iris Model Service

## Local
```bash
python train.py                 # produce models/<version>/model.joblib
docker compose up --build       # service on :8000, Prometheus on :9090
python loadtest.py              # sanity-check throughput + p99
```

## AWS (ECS Fargate)
1. **Build & push** the image to ECR:
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin $ECR
   docker build -t $ECR/iris-model:v2 .
   docker push $ECR/iris-model:v2
   ```
2. **Task definition**: container port 8000; CPU/memory sized from the load test;
   `PYTHONUNBUFFERED=1`.
3. **Service**: behind an Application Load Balancer.
   - Target-group health check path: `/readyz` (NOT `/healthz`) so traffic only
     reaches instances whose model is loaded.
   - Autoscale on CPU and on the p99 of `predict_seconds` from Prometheus.
4. **Rollback**: deploy a task def pinned to the previous image tag, OR flip the
   `ACTIVE` pointer to the prior version and redeploy. Old artifacts remain.
5. **Scaling artifacts**: for large models, fetch from S3 at startup instead of
   baking them into the image, so a model swap doesn't require an image rebuild.
