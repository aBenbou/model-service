# model-service/server.py
import os
import uuid
import yaml
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import dotenv_values
from src.config import get_config_for_endpoint, get_endpoints_for_model
from src.sagemaker.resources import list_sagemaker_endpoints, get_sagemaker_endpoint
from src.sagemaker.delete_model import delete_sagemaker_model
from src.sagemaker.create_model import deploy_model
from src.sagemaker.search_jumpstart_models import search_sagemaker_jumpstart_model
from src.sagemaker.query_endpoint import make_query_request
from src.schemas.query import Query, ChatCompletion, QueryParameters
from src.schemas.deployment import Deployment, Destination
from src.schemas.model import Model, ModelSource
from src.session import get_sagemaker_session
from src.huggingface.hf_hub_api import get_hf_task
from src.utils.model_utils import get_unique_endpoint_name
from litellm import completion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("model_manager.log")]
)
logger = logging.getLogger("model_manager")

# Set AWS region environment variable
os.environ["AWS_REGION_NAME"] = get_sagemaker_session().boto_session.region_name

# Create FastAPI app
app = FastAPI(
    title="Model Manager API",
    description="API for deploying and managing AI models on AWS SageMaker",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Deployment status tracking
deployment_status = {}

# Custom exceptions
class NotDeployedException(Exception):
    pass

# API Schemas
class DeployModelRequest(BaseModel):
    config_path: Optional[str] = None
    model_source: Optional[str] = None
    model_id: Optional[str] = None
    instance_type: Optional[str] = "ml.m5.xlarge"
    instance_count: Optional[int] = 1
    num_gpus: Optional[int] = None
    quantization: Optional[str] = None
    
class DeleteEndpointRequest(BaseModel):
    endpoint_names: List[str]

class HuggingFaceModelSearchRequest(BaseModel):
    query: str
    
class SageMakerModelSearchRequest(BaseModel):
    framework: str

# API endpoints for endpoint management
@app.get("/endpoints", tags=["Endpoints"])
def list_endpoints():
    """List all active SageMaker endpoints"""
    try:
        endpoints = list_sagemaker_endpoints()
        return {"status": "success", "endpoints": endpoints}
    except Exception as e:
        logger.error(f"Error listing endpoints: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list endpoints: {str(e)}")

@app.get("/endpoint/{endpoint_name}", tags=["Endpoints"])
def get_endpoint(endpoint_name: str):
    """Get information about a specific endpoint"""
    endpoint = get_sagemaker_endpoint(endpoint_name)
    if not endpoint:
        raise HTTPException(status_code=404, detail=f"Endpoint {endpoint_name} not found")
    return endpoint

@app.delete("/endpoints", tags=["Endpoints"])
def delete_endpoints(request: DeleteEndpointRequest):
    """Delete one or more endpoints"""
    try:
        delete_sagemaker_model(request.endpoint_names)
        return {
            "status": "success", 
            "message": f"Deleted endpoints: {request.endpoint_names}"
        }
    except Exception as e:
        logger.error(f"Error deleting endpoints: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete endpoints: {str(e)}")

# API endpoints for model deployment
@app.post("/deploy", tags=["Deployment"])
async def deploy_model_endpoint(
    request: DeployModelRequest,
    background_tasks: BackgroundTasks
):
    """Deploy a model (either from YAML or direct parameters)"""
    deployment_id = str(uuid.uuid4())
    deployment_status[deployment_id] = {
        "status": "pending",
        "start_time": datetime.now().isoformat(),
        "message": "Deployment initiated"
    }
    
    try:
        if request.config_path:
            # Deploy from YAML file
            background_tasks.add_task(
                deploy_from_yaml, 
                deployment_id, 
                request.config_path
            )
        else:
            # Deploy directly from parameters
            background_tasks.add_task(
                deploy_from_params,
                deployment_id,
                request.model_source,
                request.model_id,
                request.instance_type,
                request.instance_count,
                request.num_gpus,
                request.quantization
            )
        
        return {
            "status": "deployment_initiated", 
            "deployment_id": deployment_id,
            "message": "Model deployment has been initiated"
        }
    except Exception as e:
        logger.error(f"Error starting deployment: {str(e)}")
        deployment_status[deployment_id] = {
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "message": f"Deployment failed to start: {str(e)}"
        }
        raise HTTPException(status_code=500, detail=f"Failed to start deployment: {str(e)}")

@app.get("/deploy/{deployment_id}", tags=["Deployment"])
def get_deployment_status(deployment_id: str):
    """Get the status of a deployment"""
    if deployment_id not in deployment_status:
        raise HTTPException(status_code=404, detail="Deployment ID not found")
    
    return deployment_status[deployment_id]

# API endpoints for model querying
@app.post("/endpoint/{endpoint_name}/query", tags=["Query"])
def query_endpoint(endpoint_name: str, query: Query):
    """Query a model for inference"""
    try:
        config = get_config_for_endpoint(endpoint_name)
        if not config:
            raise HTTPException(status_code=404, detail=f"Endpoint {endpoint_name} not found")
        
        if query.context is None:
            query.context = ''

        # Support multi-model endpoints
        config = (config.deployment, config.models[0])
        result = make_query_request(endpoint_name, query, config)
        return result
    except Exception as e:
        logger.error(f"Error querying endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to query endpoint: {str(e)}")

@app.post("/chat/completions", tags=["Query"])
def chat_completion(chat_completion: ChatCompletion):
    """OpenAI-compatible chat completions endpoint"""
    try:
        model_id = chat_completion.model

        # Validate model is for completion tasks
        endpoints = get_endpoints_for_model(model_id)
        if len(endpoints) == 0:
            raise NotDeployedException("Model not deployed")

        messages = chat_completion.messages
        # Currently using the first available endpoint
        endpoint_name = endpoints[0].deployment.endpoint_name

        res = completion(
            model=f"sagemaker/{endpoint_name}",
            messages=messages,
            temperature=0.9,
            hf_model_name=model_id,
        )

        return res
    except NotDeployedException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat completion: {str(e)}")

# API endpoints for model discovery
@app.get("/models/huggingface", tags=["Models"])
def search_huggingface_models(q: Optional[str] = None):
    """Search Hugging Face models"""
    # This would implement functionality to search for models on Hugging Face
    # For now returning a placeholder
    return {"status": "not_implemented", "message": "HuggingFace model search to be implemented"}

@app.get("/models/sagemaker", tags=["Models"])
def list_sagemaker_models(framework: Optional[str] = None):
    """List SageMaker JumpStart models by framework"""
    try:
        if not framework:
            framework = "huggingface"  # Default framework
        
        filter_value = f"framework == {framework}"
        models = search_sagemaker_jumpstart_model()
        
        return {"status": "success", "models": models}
    except Exception as e:
        logger.error(f"Error listing SageMaker models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/config/{endpoint_name}", tags=["Configuration"])
def get_endpoint_config(endpoint_name: str):
    """Get the configuration for a deployed endpoint"""
    config = get_config_for_endpoint(endpoint_name)
    if not config:
        raise HTTPException(status_code=404, detail="Endpoint configuration not found")
    
    # Convert the config to a dict for JSON serialization
    return {
        "deployment": config.deployment.model_dump(),
        "models": [model.model_dump() for model in config.models]
    }

# Helper functions for background tasks
async def deploy_from_yaml(deployment_id: str, config_path: str):
    """Deploy a model from a YAML configuration file"""
    try:
        # Update status
        deployment_status[deployment_id]["message"] = "Loading YAML configuration"
        
        # Load YAML configuration
        with open(config_path) as config_file:
            configuration = yaml.safe_load(config_file)
            deployment = configuration['deployment']
            model = configuration['models'][0]
        
        # Update status
        deployment_status[deployment_id]["message"] = "Starting model deployment"
        
        # Set AWS region from environment if not specified
        region_name = os.environ.get("AWS_REGION_NAME") or "us-east-1"
        # Set environment variable for any subprocess or library that might use it
        os.environ["AWS_REGION_NAME"] = region_name
        # Also set the standard AWS environment variable
        os.environ["AWS_DEFAULT_REGION"] = region_name
        
        logger.info(f"Using AWS region: {region_name} for deployment")
        
        # Deploy model
        predictor = deploy_model(deployment, model)
        
        # Update status
        deployment_status[deployment_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "message": f"Model deployed successfully at endpoint {predictor.endpoint_name}",
            "endpoint_name": predictor.endpoint_name
        })
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        deployment_status[deployment_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "message": f"Deployment failed: {str(e)}"
        })

async def deploy_from_params(
    deployment_id: str,
    model_source: str,
    model_id: str,
    instance_type: str,
    instance_count: int,
    num_gpus: Optional[int],
    quantization: Optional[str]
):
    """Deploy a model from API parameters"""
    try:
        # Update status
        deployment_status[deployment_id]["message"] = "Preparing deployment configuration"
        
        # Set AWS region from environment if not specified
        region_name = os.environ.get("AWS_REGION_NAME") or "us-east-1"
        # Set environment variable for any subprocess or library that might use it
        os.environ["AWS_REGION_NAME"] = region_name
        # Also set the standard AWS environment variable
        os.environ["AWS_DEFAULT_REGION"] = region_name
        
        logger.info(f"Using AWS region: {region_name} for deployment")
        
        # Create model object
        model = Model(
            id=model_id,
            source=ModelSource(model_source)
        )
        
        # If it's a Hugging Face model, get the task
        if model_source == "huggingface":
            task = get_hf_task(model)
            model.task = task
        
        # Create endpoint name
        endpoint_name = get_unique_endpoint_name(model_id)
        
        # Create deployment object
        deployment = Deployment(
            destination=Destination.AWS,
            endpoint_name=endpoint_name,
            instance_type=instance_type,
            instance_count=instance_count,
            num_gpus=num_gpus,
            quantization=quantization
        )
        
        # Update status
        deployment_status[deployment_id]["message"] = "Starting model deployment"
        
        # Deploy model
        predictor = deploy_model(deployment, model)
        
        # Update status
        deployment_status[deployment_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "message": f"Model deployed successfully at endpoint {predictor.endpoint_name}",
            "endpoint_name": predictor.endpoint_name
        })
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        deployment_status[deployment_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "message": f"Deployment failed: {str(e)}"
        })

# Health check endpoint
@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "0.1.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)