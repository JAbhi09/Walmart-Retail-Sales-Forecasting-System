# Deployment Guide

## Overview

This guide covers deploying the Walmart Retail Forecasting System to various environments.

## Table of Contents

1. [Local Docker Deployment](#local-docker-deployment)
2. [Docker Hub Deployment](#docker-hub-deployment)
3. [Production Deployment](#production-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Troubleshooting](#troubleshooting)

---

## Local Docker Deployment

### Prerequisites

- Docker Desktop installed and running
- Docker Compose installed
- 8GB RAM minimum
- 10GB free disk space

### Step 1: Prepare Environment

```bash
# Clone repository
git clone https://github.com/yourusername/walmart-forecasting.git
cd walmart-forecasting

# Create environment file
cp .env.example .env
```

### Step 2: Configure .env File

Edit `.env` with your credentials:

```bash
DB_PASSWORD=your_secure_password
GEMINI_API_KEY=your_gemini_api_key
```

### Step 3: Build and Start Services

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app
```

### Step 4: Access Services

- **Dashboard**: http://localhost:8501
- **MLflow**: http://localhost:5000
- **PostgreSQL**: localhost:5432

### Step 5: Initialize Data

```bash
# Load initial data
docker-compose exec app python data_pipeline/data_loader.py

# Train initial model
docker-compose exec app python models/trainer.py
```

---

## Docker Hub Deployment

### Step 1: Build Image

```bash
# Build the image
docker build -t walmart-forecasting:latest .

# Test locally
docker run -p 8501:8501 \
  -e DB_PASSWORD=test \
  -e GEMINI_API_KEY=test \
  walmart-forecasting:latest
```

### Step 2: Tag for Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag image
docker tag walmart-forecasting:latest yourusername/walmart-forecasting:latest
docker tag walmart-forecasting:latest yourusername/walmart-forecasting:v1.0.0
```

### Step 3: Push to Docker Hub

```bash
# Push latest tag
docker push yourusername/walmart-forecasting:latest

# Push version tag
docker push yourusername/walmart-forecasting:v1.0.0
```

### Step 4: Pull and Run from Docker Hub

```bash
# Pull image
docker pull yourusername/walmart-forecasting:latest

# Run container
docker run -d \
  --name walmart-app \
  -p 8501:8501 \
  -e DB_HOST=your_db_host \
  -e DB_PASSWORD=your_password \
  -e GEMINI_API_KEY=your_key \
  yourusername/walmart-forecasting:latest
```

---

## Production Deployment

### AWS ECS/Fargate

#### 1. Create ECR Repository

```bash
# Create repository
aws ecr create-repository --repository-name walmart-forecasting

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com
```

#### 2. Build and Push

```bash
# Build for AWS
docker build -t walmart-forecasting .

# Tag for ECR
docker tag walmart-forecasting:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/walmart-forecasting:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/walmart-forecasting:latest
```

#### 3. Create Task Definition

```json
{
  "family": "walmart-forecasting",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "app",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/walmart-forecasting:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "DB_HOST", "value": "your-rds-endpoint"},
        {"name": "DB_NAME", "value": "walmart_retail"}
      ],
      "secrets": [
        {
          "name": "DB_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:db-password"
        },
        {
          "name": "GEMINI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:gemini-key"
        }
      ]
    }
  ]
}
```

#### 4. Create Service

```bash
aws ecs create-service \
  --cluster production \
  --service-name walmart-forecasting \
  --task-definition walmart-forecasting \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Google Cloud Run

```bash
# Build and submit
gcloud builds submit --tag gcr.io/<project-id>/walmart-forecasting

# Deploy
gcloud run deploy walmart-forecasting \
  --image gcr.io/<project-id>/walmart-forecasting \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DB_HOST=<cloud-sql-ip> \
  --set-secrets DB_PASSWORD=db-password:latest,GEMINI_API_KEY=gemini-key:latest \
  --memory 2Gi \
  --cpu 2
```

### Azure Container Instances

```bash
# Create resource group
az group create --name walmart-rg --location eastus

# Build and push to ACR
az acr build --registry <registry-name> \
  --image walmart-forecasting:latest .

# Deploy
az container create \
  --resource-group walmart-rg \
  --name walmart-forecasting \
  --image <registry-name>.azurecr.io/walmart-forecasting:latest \
  --cpu 2 --memory 4 \
  --ports 8501 \
  --environment-variables \
    DB_HOST=<db-host> \
    DB_NAME=walmart_retail \
  --secure-environment-variables \
    DB_PASSWORD=<password> \
    GEMINI_API_KEY=<key>
```

### Kubernetes

#### 1. Create Deployment

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
        image: yourusername/walmart-forecasting:latest
        ports:
        - containerPort: 8501
        env:
        - name: DB_HOST
          value: postgres-service
        - name: DB_NAME
          value: walmart_retail
        envFrom:
        - secretRef:
            name: app-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

#### 2. Create Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: walmart-forecasting
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8501
  selector:
    app: walmart-forecasting
```

#### 3. Deploy

```bash
# Create secrets
kubectl create secret generic app-secrets \
  --from-literal=DB_PASSWORD=<password> \
  --from-literal=GEMINI_API_KEY=<key>

# Deploy
kubectl apply -f k8s/
```

---

## Environment Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DB_HOST` | Database hostname | `localhost` or `db` |
| `DB_PORT` | Database port | `5432` |
| `DB_NAME` | Database name | `walmart_retail` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | `secure_password` |
| `GEMINI_API_KEY` | Google Gemini API key | `AIza...` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://mlflow:5000` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENVIRONMENT` | Environment name | `development` |
| `MODEL_VERSION` | Model version to use | `production` |

---

## Health Checks

### Application Health

```bash
# Check Streamlit health
curl http://localhost:8501/_stcore/health

# Check database connection
docker-compose exec app python -c "from database.db_manager import DatabaseManager; print(DatabaseManager().test_connection())"
```

### Service Status

```bash
# Check all services
docker-compose ps

# View logs
docker-compose logs -f

# Check resource usage
docker stats
```

---

## Backup and Restore

### Database Backup

```bash
# Backup database
docker-compose exec db pg_dump -U postgres walmart_retail > backup.sql

# Restore database
docker-compose exec -T db psql -U postgres walmart_retail < backup.sql
```

### Model Backup

```bash
# Backup MLflow artifacts
docker-compose exec mlflow tar -czf /tmp/mlflow-backup.tar.gz /mlflow/artifacts

# Copy to host
docker cp walmart_mlflow:/tmp/mlflow-backup.tar.gz ./mlflow-backup.tar.gz
```

---

## Scaling

### Horizontal Scaling (Multiple Instances)

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      replicas: 3
```

### Vertical Scaling (More Resources)

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

---

## Monitoring

### Logs

```bash
# View application logs
docker-compose logs -f app

# View database logs
docker-compose logs -f db

# Export logs
docker-compose logs > logs.txt
```

### Metrics

```bash
# Container stats
docker stats

# Disk usage
docker system df
```

---

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker-compose logs app

# Check environment variables
docker-compose config

# Rebuild image
docker-compose build --no-cache app
```

#### Database Connection Failed

```bash
# Check database is running
docker-compose ps db

# Test connection
docker-compose exec db psql -U postgres -c "SELECT 1"

# Check network
docker network inspect retail_project_walmart_network
```

#### Out of Memory

```bash
# Increase memory limit
# Edit docker-compose.yml
services:
  app:
    mem_limit: 4g
```

#### Port Already in Use

```bash
# Find process using port
netstat -ano | findstr :8501  # Windows
lsof -i :8501                 # Linux/Mac

# Change port in docker-compose.yml
ports:
  - "8502:8501"
```

---

## Security Best Practices

1. **Never commit .env files** - Use .env.example as template
2. **Use secrets management** - AWS Secrets Manager, Azure Key Vault, etc.
3. **Rotate credentials regularly** - Database passwords, API keys
4. **Use HTTPS in production** - Configure reverse proxy (nginx, Traefik)
5. **Limit database access** - Use firewall rules and VPC
6. **Scan images for vulnerabilities** - Use Docker Scout or Trivy

---

## Maintenance

### Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

### Update Dependencies

```bash
# Update requirements.txt
pip install --upgrade -r requirements.txt

# Rebuild image
docker-compose build app
```

### Clean Up

```bash
# Remove stopped containers
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Clean up Docker system
docker system prune -a
```

---

## Next Steps

1. ✅ Deploy locally with Docker Compose
2. ✅ Test all services are running
3. ✅ Load sample data
4. ✅ Train initial model
5. ✅ Push to Docker Hub
6. ✅ Deploy to production environment
7. ✅ Set up monitoring and alerts
8. ✅ Configure backups

For CI/CD setup, see [CI_CD_GUIDE.md](CI_CD_GUIDE.md)
