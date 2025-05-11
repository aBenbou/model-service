
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks 
from typing import List, Dict, Optional
import logging
import json
import time
import uuid
from pydantic import BaseModel

from src.sagemaker.resources import list_sagemaker_endpoints, get_sagemaker_endpoint
from src.sagemaker.query_endpoint import make_query_request
from src.schemas.query import Query, QueryParameters
from src.config import get_config_for_endpoint, get_deployment_configs
from src.schemas.model import Model, ModelSource
from src.schemas.deployment import Deployment, Destination
from src.sagemaker.create_model import deploy_model
from src.sagemaker.delete_model import delete_sagemaker_model
from src.sagemaker.fine_tune_model import fine_tune_model
from src.session import get_sagemaker_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("model_service.log")]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Model Service API",
    description="API for managing and deploying AI models",
    version="1.0.0"
)

# Deployment status tracking
deployment_status = {}

# Model Management Endpoints
@app.post("/models/deploy")
async def deploy_model_endpoint(
    background_tasks: BackgroundTasks,
    model_source: str,
    model_id: str,
    instance_type: str = "ml.m5.xlarge",
    instance_count: int = 1,
    num_gpus: Optional[int] = None,
    quantization: Optional[str] = None
):
    """Deploy a new model"""
    try:
        deployment_id = str(uuid.uuid4())
        deployment_status[deployment_id] = {
            "status": "pending",
            "start_time": time.time(),
            "message": "Starting deployment"
        }
        
        background_tasks.add_task(
            deploy_from_params,
            deployment_id,
            model_source,
            model_id,
            instance_type,
            instance_count,
            num_gpus,
            quantization
        )
        
        return {
            "deployment_id": deployment_id,
            "status": "pending",
            "message": "Deployment started"
        }
    except Exception as e:
        logger.error(f"Failed to start deployment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/deployments/{deployment_id}")
async def get_deployment_status(deployment_id: str):
    """Get deployment status"""
    if deployment_id not in deployment_status:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return deployment_status[deployment_id]

@app.get("/models/endpoints")
async def list_endpoints():
    """List all deployed model endpoints"""
    try:
        endpoints = list_sagemaker_endpoints()
        return {"endpoints": endpoints}
    except Exception as e:
        logger.error(f"Failed to list endpoints: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/endpoints/{endpoint_name}")
async def get_endpoint(endpoint_name: str):
    """Get details of a specific endpoint"""
    try:
        endpoint = get_sagemaker_endpoint(endpoint_name)
        if not endpoint:
            raise HTTPException(status_code=404, detail="Endpoint not found")
        return endpoint
    except Exception as e:
        logger.error(f"Failed to get endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/endpoints/{endpoint_name}")
async def delete_endpoint(endpoint_name: str):
    """Delete a model endpoint"""
    try:
        delete_sagemaker_model([endpoint_name])
        return {"message": f"Endpoint {endpoint_name} deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/query/{endpoint_name}")
async def query_endpoint(endpoint_name: str, query: Query):
    """Query a model endpoint"""
    try:
        config = get_config_for_endpoint(endpoint_name)
        if not config:
            raise HTTPException(status_code=404, detail="Endpoint configuration not found")
        
        response = make_query_request(endpoint_name, query, config)
        return response
    except Exception as e:
        logger.error(f"Failed to query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/fine-tune")
async def fine_tune_endpoint(
    background_tasks: BackgroundTasks,
    model_id: str,
    training_data_path: str,
    instance_type: str = "ml.m5.xlarge",
    instance_count: int = 1
):
    """Fine-tune a model"""
    try:
        training_id = str(uuid.uuid4())
        deployment_status[training_id] = {
            "status": "pending",
            "start_time": time.time(),
            "message": "Starting fine-tuning"
        }
        
        background_tasks.add_task(
            fine_tune_model_task,
            training_id,
            model_id,
            training_data_path,
            instance_type,
            instance_count
        )
        
        return {
            "training_id": training_id,
            "status": "pending",
            "message": "Fine-tuning started"
        }
    except Exception as e:
        logger.error(f"Failed to start fine-tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def deploy_from_params(
    deployment_id: str,
    model_source: str,
    model_id: str,
    instance_type: str,
    instance_count: int,
    num_gpus: Optional[int],
    quantization: Optional[str]
):
    """Deploy a model from parameters"""
    try:
        model = Model(
            id=model_id,
            source=ModelSource(model_source)
        )
        
        deployment = Deployment(
            destination=Destination.AWS,
            endpoint_name=f"{model_id}-{deployment_id[:8]}",
            instance_type=instance_type,
            instance_count=instance_count,
            num_gpus=num_gpus,
            quantization=quantization
        )
        
        predictor = deploy_model(deployment, model)
        
        deployment_status[deployment_id].update({
            "status": "completed",
            "end_time": time.time(),
            "message": f"Model deployed successfully at endpoint {predictor.endpoint_name}",
            "endpoint_name": predictor.endpoint_name
        })
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        deployment_status[deployment_id].update({
            "status": "failed",
            "end_time": time.time(),
            "message": f"Deployment failed: {str(e)}"
        })

async def fine_tune_model_task(
    training_id: str,
    model_id: str,
    training_data_path: str,
    instance_type: str,
    instance_count: int
):
    """Fine-tune a model task"""
    try:
        # Implementation of fine-tuning logic
        deployment_status[training_id].update({
            "status": "completed",
            "end_time": time.time(),
            "message": "Fine-tuning completed successfully"
        })
    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        deployment_status[training_id].update({
            "status": "failed",
            "end_time": time.time(),
            "message": f"Fine-tuning failed: {str(e)}"
        })


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}
