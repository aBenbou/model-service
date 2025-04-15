# Model-Service

This Service is a REST API for deploying and managing AI models on AWS SageMaker. It provides a streamlined interface to deploy, manage, and query open source AI models without dealing with the complexities of AWS SageMaker configuration.

## About Model Manager Service

It simplifies the process of deploying an open source AI model to your own cloud and lets you deploy open source AI models through a REST API.

Choose a model from Hugging Face or SageMaker, and the Service will spin up a SageMaker instance with a ready-to-query endpoint in minutes.

## Features

- **Model Deployment**: Deploy models from Hugging Face, SageMaker, or custom sources to AWS SageMaker
- **Endpoint Management**: List, inspect, and delete SageMaker endpoints
- **Model Querying**: Query deployed models for inference via REST API
- **Asynchronous Operations**: Long-running deployments are handled asynchronously with status tracking
- **OpenAI-compatible Interface**: Chat completion endpoint compatible with OpenAI's API
- **YAML Configuration**: Support for YAML-based deployment specifications
- **Multiple Deployment Methods**: Support for Hugging Face models, SageMaker JumpStart models, and custom models

## Architecture

Service consists of:

- **FastAPI Backend**: Provides a RESTful API for all operations
- **AWS Integration**: Manages AWS SageMaker resources
- **Docker Containerization**: Enables easy deployment and isolation

## Prerequisites

- Docker and Docker Compose
- AWS account with SageMaker access
- Quota for AWS SageMaker instances (by default, you get 2 instances of ml.m5.xlarge for free)
- (Optional) Hugging Face account and token for accessing gated models (e.g., Llama2)

## Installation and Setup

### Step 1: Set up AWS and SageMaker

To get started, you'll need an AWS account which you can create at https://aws.amazon.com/. Then you'll need to create access keys for SageMaker with appropriate permissions.

You'll need:
- AWS Access Key ID
- AWS Secret Access Key
- AWS Region (default is us-east-1)

### Step 2: Clone the Repository

```bash
git clone https://github.com/aBenbou/model-service.git
cd model-service
```

### Step 3: Run the Setup Script

```bash
bash service-setup.sh
```

This will:
- Configure AWS credentials
- Set up a SageMaker execution role
- Create necessary directories
- (Optional) Add your Hugging Face token

### Step 4: Start the Service

```bash
docker-compose up -d
```

The API will be available at: http://localhost:8000
Swagger documentation: http://localhost:8000/docs

## Usage

### Deploying Models

There are two ways to deploy models:

#### Option 1: Direct API Request

```bash
curl -X POST http://localhost:8000/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "model_source": "huggingface",
    "model_id": "google-bert/bert-base-uncased",
    "instance_type": "ml.m5.xlarge",
    "instance_count": 1
  }'
```

This returns a deployment ID that you can use to check the status.

#### Option 2: Using a YAML Configuration File

First, create a YAML file in the `configs/` directory:

```yaml
# configs/bert-base.yaml
deployment: !Deployment
  destination: aws
  instance_type: ml.m5.xlarge
  instance_count: 1
  num_gpus: null
  quantization: null

models:
- !Model
  id: google-bert/bert-base-uncased
  source: huggingface
  task: fill-mask
```

Then deploy using the config file:

```bash
curl -X POST http://localhost:8000/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "./configs/bert-base.yaml"
  }'
```

### Checking Deployment Status

```bash
curl -X GET http://localhost:8000/deploy/{deployment_id}
```

### Listing Endpoints

```bash
curl -X GET http://localhost:8000/endpoints
```

### Querying Models

The query format depends on the type of model:

#### Fill Mask Models (like BERT)

```bash
curl -X POST http://localhost:8000/endpoint/{endpoint_name}/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Paris is the capital of [MASK]."
  }'
```

#### Question Answering Models

```bash
curl -X POST http://localhost:8000/endpoint/{endpoint_name}/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "context": "France is a country in Western Europe. Its capital is Paris."
  }'
```

#### Text Generation Models

```bash
curl -X POST http://localhost:8000/endpoint/{endpoint_name}/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Write a poem about artificial intelligence",
    "parameters": {
      "max_new_tokens": 250,
      "temperature": 0.7,
      "top_p": 0.9
    }
  }'
```

#### OpenAI-Compatible Chat Completions

```bash
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ]
  }'
```

### Deleting Endpoints

```bash
curl -X DELETE http://localhost:8000/endpoints \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint_names": ["endpoint-name-1", "endpoint-name-2"]
  }'
```

## Supported Models

If you're using the `ml.m5.xlarge` instance type, here are some small models that work well:

### Hugging Face Models

1. **google-bert/bert-base-uncased**
   - Type: Fill Mask
   - Query format: Text with `[MASK]` token

2. **sentence-transformers/all-MiniLM-L6-v2**
   - Type: Feature extraction
   - Query format: Text string

3. **deepset/roberta-base-squad2**
   - Type: Question answering
   - Query format: Question and context

### SageMaker JumpStart Models

1. **huggingface-tc-bert-base-cased**
   - Type: Text classification
   - Query format: Text string

2. **huggingface-eqa-bert-base-cased**
   - Type: Extractive question answering
   - Query format: Question and context

## API Documentation

For detailed API documentation, see the [API Documentation](API_DOCUMENTATION.md) or visit the Swagger UI at http://localhost:8000/docs when the service is running.

## Troubleshooting

### Common Issues

1. **AWS Credential Errors**
   - Ensure your AWS credentials in `.env` are correct and have appropriate permissions

2. **Role ARN Format Errors**
   - Make sure you're using a proper SageMaker execution role ARN (`:role/`) and not a user ARN (`:user/`)

3. **Hugging Face Authentication Issues**
   - If using gated models, check that your Hugging Face token is valid

### Logs

To view service logs:

```bash
docker-compose logs -f
```

## Development

### Project Structure

```
model-service/
├── configs/               # YAML configuration files
├── models/                # Local model storage
├── scripts/               # Utility scripts
├── src/                   # Source code
│   ├── huggingface/       # Hugging Face integration
│   ├── sagemaker/         # SageMaker integration
│   ├── schemas/           # Data models
│   └── utils/             # Utility functions
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── model_manager.py       # CLI entry point
├── requirements.txt       # Python dependencies
├── server.py              # FastAPI server
└── service-setup.sh       # Setup script
```

### Local Development

For development without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the setup script
bash service-setup.sh

# Start the API server in development mode
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

## Known Issues

- Querying within Model Manager currently only works with text-based models
- Model versions are static
- Deleting a model is not instant, it may show up briefly after it was queued for deletion
- Deploying the same model within the same minute will break