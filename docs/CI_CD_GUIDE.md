# CI/CD Pipeline Guide for Walmart Retail Forecasting System

## Overview

This guide provides step-by-step instructions for setting up Continuous Integration and Continuous Deployment (CI/CD) pipelines for the retail forecasting system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [GitHub Actions Setup](#github-actions-setup)
3. [GitLab CI Setup](#gitlab-ci-setup)
4. [Environment Variables & Secrets](#environment-variables--secrets)
5. [Testing Strategy](#testing-strategy)
6. [Deployment Options](#deployment-options)
7. [Monitoring & Alerts](#monitoring--alerts)

---

## Prerequisites

### Required Accounts
- GitHub or GitLab account
- Docker Hub account (for container registry)
- Cloud provider account (AWS/GCP/Azure) for deployment
- PostgreSQL database (production)

### Required Secrets
- `DB_PASSWORD` - PostgreSQL password
- `GEMINI_API_KEY` - Google Gemini API key
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password
- Cloud provider credentials (AWS/GCP/Azure)

---

## GitHub Actions Setup

### 1. Create Workflow Directory

```bash
mkdir -p .github/workflows
```

### 2. Create CI Workflow

Create `.github/workflows/ci.yml`:

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: walmart_retail_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-mock
      
      - name: Set up environment variables
        run: |
          echo "DB_HOST=localhost" >> $GITHUB_ENV
          echo "DB_PORT=5432" >> $GITHUB_ENV
          echo "DB_NAME=walmart_retail_test" >> $GITHUB_ENV
          echo "DB_USER=postgres" >> $GITHUB_ENV
          echo "DB_PASSWORD=postgres" >> $GITHUB_ENV
          echo "GEMINI_API_KEY=${{ secrets.GEMINI_API_KEY }}" >> $GITHUB_ENV
      
      - name: Initialize database
        run: |
          python scripts/setup_database.py
        env:
          DB_PASSWORD: postgres
      
      - name: Run unit tests
        run: |
          pytest tests/ -v --cov=. --cov-report=xml --cov-report=term
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
  
  lint:
    name: Code Quality
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install linting tools
        run: |
          pip install flake8 black isort mypy
      
      - name: Run Black
        run: black --check .
      
      - name: Run isort
        run: isort --check-only .
      
      - name: Run Flake8
        run: flake8 . --max-line-length=100 --exclude=venv,__pycache__
      
      - name: Run MyPy
        run: mypy . --ignore-missing-imports || true
```

### 3. Create CD Workflow

Create `.github/workflows/cd.yml`:

```yaml
name: CD Pipeline

on:
  push:
    branches: [ main ]
    tags:
      - 'v*'

jobs:
  build-and-push:
    name: Build and Push Docker Image
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ secrets.DOCKER_USERNAME }}/walmart-forecasting
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
  
  deploy-staging:
    name: Deploy to Staging
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Deploy to staging
        run: |
          echo "Deploying to staging environment..."
          # Add your deployment script here
          # Example: kubectl apply -f k8s/staging/
      
      - name: Run smoke tests
        run: |
          echo "Running smoke tests..."
          # Add smoke test commands
  
  deploy-production:
    name: Deploy to Production
    needs: build-and-push
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    environment: production
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Deploy to production
        run: |
          echo "Deploying to production environment..."
          # Add your production deployment script
      
      - name: Run health checks
        run: |
          echo "Running health checks..."
          # Add health check commands
```

---

## GitLab CI Setup

Create `.gitlab-ci.yml`:

```yaml
stages:
  - test
  - build
  - deploy

variables:
  POSTGRES_DB: walmart_retail_test
  POSTGRES_USER: postgres
  POSTGRES_PASSWORD: postgres
  POSTGRES_HOST_AUTH_METHOD: trust

test:
  stage: test
  image: python:3.11
  
  services:
    - postgres:16
  
  variables:
    DB_HOST: postgres
    DB_PORT: "5432"
    DB_NAME: walmart_retail_test
    DB_USER: postgres
    DB_PASSWORD: postgres
  
  before_script:
    - pip install --upgrade pip
    - pip install -r requirements.txt
    - pip install pytest pytest-cov pytest-mock
  
  script:
    - python scripts/setup_database.py
    - pytest tests/ -v --cov=. --cov-report=xml --cov-report=term
  
  coverage: '/(?i)total.*? (100(?:\.0+)?\%|[1-9]?\d(?:\.\d+)?\%)$/'
  
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

lint:
  stage: test
  image: python:3.11
  
  before_script:
    - pip install flake8 black isort
  
  script:
    - black --check .
    - isort --check-only .
    - flake8 . --max-line-length=100 --exclude=venv,__pycache__

build:
  stage: build
  image: docker:24
  
  services:
    - docker:24-dind
  
  before_script:
    - docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD
  
  script:
    - docker build -t $DOCKER_USERNAME/walmart-forecasting:$CI_COMMIT_SHA .
    - docker push $DOCKER_USERNAME/walmart-forecasting:$CI_COMMIT_SHA
    - |
      if [ "$CI_COMMIT_BRANCH" == "main" ]; then
        docker tag $DOCKER_USERNAME/walmart-forecasting:$CI_COMMIT_SHA $DOCKER_USERNAME/walmart-forecasting:latest
        docker push $DOCKER_USERNAME/walmart-forecasting:latest
      fi
  
  only:
    - main
    - tags

deploy_staging:
  stage: deploy
  image: alpine:latest
  
  before_script:
    - apk add --no-cache curl
  
  script:
    - echo "Deploying to staging..."
    # Add deployment commands
  
  only:
    - main
  
  environment:
    name: staging
    url: https://staging.example.com

deploy_production:
  stage: deploy
  image: alpine:latest
  
  before_script:
    - apk add --no-cache curl
  
  script:
    - echo "Deploying to production..."
    # Add deployment commands
  
  only:
    - tags
  
  when: manual
  
  environment:
    name: production
    url: https://production.example.com
```

---

## Environment Variables & Secrets

### GitHub Secrets Setup

1. Go to your repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add the following secrets:

```
GEMINI_API_KEY=your_gemini_api_key
DOCKER_USERNAME=your_docker_username
DOCKER_PASSWORD=your_docker_password
DB_PASSWORD=your_production_db_password
```

### GitLab CI/CD Variables Setup

1. Go to Settings → CI/CD → Variables
2. Add the following variables:

| Key | Value | Protected | Masked |
|-----|-------|-----------|--------|
| `GEMINI_API_KEY` | your_api_key | ✓ | ✓ |
| `DOCKER_USERNAME` | your_username | ✓ | ✗ |
| `DOCKER_PASSWORD` | your_password | ✓ | ✓ |
| `DB_PASSWORD` | your_db_password | ✓ | ✓ |

---

## Testing Strategy

### Test Stages

1. **Unit Tests** (Fast, ~2-3 minutes)
   - Database operations
   - Feature engineering
   - ML models
   - AI agents

2. **Integration Tests** (Medium, ~5-10 minutes)
   - End-to-end pipelines
   - Database consistency
   - System performance

3. **Validation Tests** (Slow, ~10-15 minutes)
   - Model accuracy
   - Data quality
   - Business logic

### Parallel Testing

```yaml
# GitHub Actions example
jobs:
  test-unit:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/test_database.py tests/test_model.py -v
  
  test-integration:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/test_integration.py -v
  
  test-validation:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/test_validation.py -v
```

---

## Deployment Options

### Option 1: Docker Compose (Simple)

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    image: your-username/walmart-forecasting:latest
    ports:
      - "8501:8501"
    environment:
      - DB_HOST=db
      - DB_PASSWORD=${DB_PASSWORD}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    depends_on:
      - db
  
  db:
    image: postgres:16
    environment:
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Deploy:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Option 2: Kubernetes (Scalable)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: walmart-forecasting
spec:
  replicas: 3
  selector:
    matchLabels:
      app: walmart-forecasting
  template:
    metadata:
      labels:
        app: walmart-forecasting
    spec:
      containers:
      - name: app
        image: your-username/walmart-forecasting:latest
        ports:
        - containerPort: 8501
        env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: db-password
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: gemini-api-key
```

Deploy:
```bash
kubectl apply -f k8s/
```

### Option 3: Cloud Platforms

#### AWS (ECS/Fargate)
```bash
# Build and push
docker build -t walmart-forecasting .
docker tag walmart-forecasting:latest <aws-account>.dkr.ecr.us-east-1.amazonaws.com/walmart-forecasting:latest
docker push <aws-account>.dkr.ecr.us-east-1.amazonaws.com/walmart-forecasting:latest

# Deploy
aws ecs update-service --cluster production --service walmart-forecasting --force-new-deployment
```

#### GCP (Cloud Run)
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/<project-id>/walmart-forecasting
gcloud run deploy walmart-forecasting \
  --image gcr.io/<project-id>/walmart-forecasting \
  --platform managed \
  --region us-central1 \
  --set-env-vars GEMINI_API_KEY=$GEMINI_API_KEY
```

#### Azure (Container Instances)
```bash
# Build and push
az acr build --registry <registry-name> --image walmart-forecasting:latest .

# Deploy
az container create \
  --resource-group myResourceGroup \
  --name walmart-forecasting \
  --image <registry-name>.azurecr.io/walmart-forecasting:latest \
  --environment-variables GEMINI_API_KEY=$GEMINI_API_KEY
```

---

## Monitoring & Alerts

### Health Check Endpoint

Add to your Streamlit app:

```python
# dashboard/health.py
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### GitHub Actions Monitoring

```yaml
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Prometheus Metrics (Optional)

```python
# Add to your app
from prometheus_client import Counter, Histogram, start_http_server

prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

# Use in your code
prediction_counter.inc()
```

---

## Best Practices

### 1. Branch Protection Rules
- Require pull request reviews
- Require status checks to pass
- Require branches to be up to date

### 2. Semantic Versioning
```bash
# Tag releases
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### 3. Database Migrations
```yaml
- name: Run migrations
  run: |
    python scripts/migrate_database.py
```

### 4. Rollback Strategy
```bash
# Quick rollback
kubectl rollout undo deployment/walmart-forecasting
# or
docker-compose down && docker-compose up -d --force-recreate
```

### 5. Environment-Specific Configs
```
config/
├── development.yaml
├── staging.yaml
└── production.yaml
```

---

## Troubleshooting

### Common Issues

**Issue**: Tests fail in CI but pass locally
- **Solution**: Ensure environment variables are set correctly in CI

**Issue**: Docker build fails
- **Solution**: Check Dockerfile syntax and base image availability

**Issue**: Database connection timeout
- **Solution**: Increase health check timeout in service configuration

**Issue**: Out of memory during tests
- **Solution**: Use `pytest-xdist` for parallel testing or increase runner memory

---

## Next Steps

1. ✅ Set up GitHub/GitLab repository
2. ✅ Add secrets to CI/CD platform
3. ✅ Create workflow files
4. ✅ Test CI pipeline with a pull request
5. ✅ Configure deployment target (Docker/K8s/Cloud)
6. ✅ Set up monitoring and alerts
7. ✅ Document deployment process for team

---

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitLab CI/CD Documentation](https://docs.gitlab.com/ee/ci/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
