# model-service/src/api.py
from fastapi import FastAPI, HTTPException, Request
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

logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

@app.get("/endpoints")
async def list_endpoints():
    """List all available model endpoints."""
    try:
        endpoints = list_sagemaker_endpoints()
        # Format endpoint data to match what Interaction Service expects exactly
        return {
            "endpoints": [
                {
                    "endpointName": endpoint["EndpointName"],
                    "status": endpoint["EndpointStatus"],
                    "instanceType": endpoint.get("InstanceType", "unknown"),
                    "creationTime": str(endpoint.get("CreationTime", ""))
                } for endpoint in endpoints
            ]
        }
    except Exception as e:
        logger.error(f"Error listing endpoints: {str(e)}")
        # Return error in the format Interaction Service expects
        return {
            "error": f"Failed to list endpoints: {str(e)}"
        }

@app.get("/endpoint/{endpoint_name}")
async def get_endpoint_details(endpoint_name: str):
    """Get details about a specific endpoint."""
    try:
        endpoint_data = get_sagemaker_endpoint(endpoint_name)
        if endpoint_data is None:
            # Return 404 in the format Interaction Service expects
            return {
                "error": f"Endpoint {endpoint_name} not found"
            }
        
        # Extract model ID from endpoint name
        model_id = endpoint_name.split('-2')[0]  # Remove timestamp suffix
        model_id = model_id.replace('--', '-')   # Convert double-dash to single-dash
        
        # Format the response to match exactly what the Interaction Service expects
        return {
            "models": [
                {
                    "id": model_id,
                    "version": "1.0"
                }
            ],
            "endpointName": endpoint_name,
            "status": endpoint_data.get('deployment', {}).get('endpointStatus', 'Active')
        }
    except Exception as e:
        logger.error(f"Error getting endpoint {endpoint_name}: {str(e)}")
        # Return error in the format Interaction Service expects
        return {
            "error": f"Failed to get endpoint details: {str(e)}"
        }

@app.post("/endpoint/{endpoint_name}/query")
async def query_endpoint(endpoint_name: str, request: Request):
    """Query a model endpoint."""
    try:
        # Parse request body
        data = await request.json()
        query_text = data.get("query")
        if not query_text:
            return {"error": "'query' field is required"}
        
        # Convert parameters format if present
        parameters = None
        if data.get("parameters"):
            parameters = QueryParameters(**data.get("parameters"))
        
        # Create Query object
        query = Query(
            query=query_text,
            context=data.get("context"),
            parameters=parameters
        )
        
        # Get config for this endpoint
        config_obj = get_config_for_endpoint(endpoint_name)
        if not config_obj:
            return {"error": f"Configuration for endpoint {endpoint_name} not found"}
        
        # Extract deployment and model from config
        if not config_obj.models:
            return {"error": "No models found in the endpoint configuration"}
            
        config = (config_obj.deployment, config_obj.models[0])
        
        # Make the query
        result = make_query_request(endpoint_name, query, config)
        return result
    except Exception as e:
        logger.error(f"Error querying endpoint {endpoint_name}: {str(e)}")
        return {"error": f"Failed to query endpoint: {str(e)}"}

@app.post("/chat/completions")
async def chat_completion(request: Request):
    """OpenAI-compatible chat completion endpoint."""
    try:
        data = await request.json()
        model_id = data.get("model")
        messages = data.get("messages", [])
        
        if not model_id or not messages:
            raise HTTPException(status_code=400, detail="'model' and 'messages' fields are required")
        
        # Find the latest user message
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user messages found")
        
        latest_user_message = user_messages[-1]["content"]
        
        # Find configs for this model
        configs = []
        for config in get_deployment_configs():
            for model in config.models:
                if model.id == model_id:
                    configs.append((config.deployment, model))
        
        if not configs:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Use the first config found
        config = configs[0]
        
        # Create a query
        query = Query(
            query=latest_user_message,
            context=json.dumps(messages[:-1])  # Previous messages as context
        )
        
        # Query the model
        response_text = make_query_request(config[0].endpoint_name, query, config)
        
        # If response is a dict with generated text, extract it
        if isinstance(response_text, dict):
            if "generated_text" in response_text:
                response_text = response_text["generated_text"]
            elif "answer" in response_text:
                response_text = response_text["answer"]
            elif "content" in response_text:
                response_text = response_text["content"]
        
        # Return in OpenAI format
        return {
            "id": f"chatcmpl-{model_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(response_text)
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate chat completion: {str(e)}")

class DeploymentRequest(BaseModel):
    model_source: str
    model_id: str
    instance_type: str
    instance_count: int = 1
    num_gpus: Optional[int] = None
    quantization: Optional[str] = None

@app.post("/deploy")
async def deploy_model_endpoint(request: DeploymentRequest):
    """Deploy a model to create a new endpoint."""
    try:
        # Map the model source string to ModelSource enum
        model_source_map = {
            "huggingface": ModelSource.HuggingFace,
            "sagemaker": ModelSource.Sagemaker,
            "custom": ModelSource.Custom
        }
        
        if request.model_source.lower() not in model_source_map:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model_source. Use one of: {', '.join(model_source_map.keys())}"
            )
        
        # Create Model object
        model = Model(
            id=request.model_id,
            source=model_source_map[request.model_source.lower()]
        )
        
        # Create Deployment object with a unique endpoint name
        endpoint_name = f"{request.model_id.replace('/', '--').replace('_', '-').replace('.', '')[:50]}-{uuid.uuid4().hex[:8]}"
        
        deployment = Deployment(
            destination=Destination.AWS,
            instance_type=request.instance_type,
            endpoint_name=endpoint_name,
            instance_count=request.instance_count,
            num_gpus=request.num_gpus,
            quantization=request.quantization
        )
        
        # Deploy the model
        logger.info(f"Deploying model {request.model_id} to endpoint {endpoint_name}")
        predictor = deploy_model(deployment, model)
        
        # Return deployment info
        return {
            "success": True,
            "endpoint_name": deployment.endpoint_name,
            "model_id": model.id,
            "instance_type": deployment.instance_type,
            "status": "deploying"
        }
    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to deploy model: {str(e)}")

@app.delete("/endpoint/{endpoint_name}")
async def delete_endpoint(endpoint_name: str):
    """Delete a deployed endpoint."""
    try:
        from src.sagemaker.delete_model import delete_sagemaker_model
        
        logger.info(f"Deleting endpoint: {endpoint_name}")
        delete_sagemaker_model([endpoint_name])
        
        return {"success": True, "message": f"Endpoint {endpoint_name} deletion initiated"}
    except Exception as e:
        logger.error(f"Error deleting endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete endpoint: {str(e)}")

@app.get("/models")
async def list_models(framework: Optional[str] = None):
    """List available models from HuggingFace or SageMaker."""
    try:
        from src.sagemaker.search_jumpstart_models import search_sagemaker_jumpstart_model
        
        models = search_sagemaker_jumpstart_model(framework)
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/deploy/{deployment_id}")
async def get_deployment_status(deployment_id: str):
    """Get the status of a model deployment."""
    try:
        from src.server import deployment_status
        
        if deployment_id not in deployment_status:
            raise HTTPException(status_code=404, detail="Deployment ID not found")
        
        return deployment_status[deployment_id]
    except Exception as e:
        logger.error(f"Error getting deployment status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get deployment status: {str(e)}")

@app.get("/models/huggingface")
async def search_huggingface_models(q: Optional[str] = None):
    """Search Hugging Face models."""
    try:
        # This would implement functionality to search for models on Hugging Face
        # For now returning a placeholder
        return {"status": "not_implemented", "message": "HuggingFace model search to be implemented"}
    except Exception as e:
        logger.error(f"Error searching HuggingFace models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search models: {str(e)}")

@app.get("/models/sagemaker")
async def list_sagemaker_models(framework: Optional[str] = None):
    """List SageMaker JumpStart models by framework."""
    try:
        from src.sagemaker.search_jumpstart_models import search_sagemaker_jumpstart_model
        
        if not framework:
            framework = "huggingface"  # Default framework
        
        models = search_sagemaker_jumpstart_model(framework)
        
        return {"status": "success", "models": models}
    except Exception as e:
        logger.error(f"Error listing SageMaker models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/config/{endpoint_name}")
async def get_endpoint_config(endpoint_name: str):
    """Get the configuration for a deployed endpoint."""
    try:
        from src.config import get_config_for_endpoint
        
        config = get_config_for_endpoint(endpoint_name)
        if not config:
            raise HTTPException(status_code=404, detail="Endpoint configuration not found")
        
        # Convert the config to a dict for JSON serialization
        return {
            "deployment": config.deployment.model_dump(),
            "models": [model.model_dump() for model in config.models]
        }
    except Exception as e:
        logger.error(f"Error getting endpoint configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get endpoint configuration: {str(e)}")

# Add compatibility for bulk deletion like in server.py
class DeleteEndpointRequest(BaseModel):
    endpoint_names: List[str]

@app.delete("/endpoints")
async def delete_endpoints(request: DeleteEndpointRequest):
    """Delete one or more endpoints."""
    try:
        from src.sagemaker.delete_model import delete_sagemaker_model
        
        delete_sagemaker_model(request.endpoint_names)
        return {
            "status": "success", 
            "message": f"Deleted endpoints: {request.endpoint_names}"
        }
    except Exception as e:
        logger.error(f"Error deleting endpoints: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete endpoints: {str(e)}")

@app.get("/models/validate/{model_id}")
async def validate_model(model_id: str, model_version: Optional[str] = None):
    """Validate if a model exists and is deployed."""
    try:
        # Convert model_id to the format used in endpoints
        model_id_for_comparison = model_id.replace('-', '--')
        
        # Get all endpoints
        endpoints = list_sagemaker_endpoints()
        
        for endpoint in endpoints:
            endpoint_name = endpoint["EndpointName"]
            # Check if the endpoint name contains the model ID
            if model_id_for_comparison in endpoint_name:
                return {
                    "valid": True,
                    "endpoint_name": endpoint_name
                }
        
        # Model not found in any endpoints
        return {
            "valid": False,
            "error": "Model not found in any active endpoints"
        }
    except Exception as e:
        logger.error(f"Error validating model {model_id}: {str(e)}")
        return {"error": f"Failed to validate model: {str(e)}"}
