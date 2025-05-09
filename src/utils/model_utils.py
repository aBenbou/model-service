# model-service/src/utils/model_utils.py
import datetime
import logging
from difflib import SequenceMatcher
from dotenv import dotenv_values
from huggingface_hub import HfApi
from src.utils.rich_utils import print_error
from src.sagemaker import SagemakerTask
from src.schemas.deployment import Deployment
from src.schemas.model import Model, ModelSource
from src.schemas.query import Query
from src.session import get_sagemaker_session
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

HUGGING_FACE_HUB_TOKEN = dotenv_values(".env").get("HUGGING_FACE_HUB_KEY")


def get_unique_endpoint_name(model_id: str, endpoint_name: str = None):
    """
    Generate a unique endpoint name
    
    Args:
        model_id: Model ID
        endpoint_name: Optional existing endpoint name
        
    Returns:
        Unique endpoint name
    """
    dt_string = datetime.datetime.now().strftime("%Y%m%d%H%M")

    if not endpoint_name:
        # Endpoint name must be < 63 characters
        model_string = model_id.replace(
            "/", "--").replace("_", "-").replace(".", "")[:50]
        return f"{model_string}-{dt_string}"
    else:
        return f"{endpoint_name[:50]}-{dt_string}"


def is_sagemaker_model(endpoint_name: str, config: Optional[Tuple[Deployment, Model]] = None) -> bool:
    """
    Check if a model is a SageMaker model
    
    Args:
        endpoint_name: Name of the endpoint
        config: Optional tuple of (deployment, model)
        
    Returns:
        True if the model is a SageMaker model
    """
    if config is not None:
        _, model = config
        task = get_sagemaker_model_and_task(model.id)['task']

        # check task for custom sagemaker models
        return model.source == ModelSource.Sagemaker or task in SagemakerTask.list()

    # fallback
    return endpoint_name.find("--") == -1


def is_custom_model(endpoint_name: str) -> bool:
    """
    Check if a model is a custom model
    
    Args:
        endpoint_name: Name of the endpoint
        
    Returns:
        True if the model is a custom model
    """
    return endpoint_name.startswith('custom')


def get_sagemaker_model_and_task(endpoint_or_model_name: str):
    """
    Get model ID and task from SageMaker endpoint or model name
    
    Args:
        endpoint_or_model_name: Name of the endpoint or model
        
    Returns:
        Dictionary with model_id and task
    """
    components = endpoint_or_model_name.split('-')
    if len(components) < 2:
        return {
            'model_id': endpoint_or_model_name,
            'task': 'text-generation'  # Default to text-generation if we can't determine task
        }
        
    framework, task = components[:2]
    model_id = '-'.join(components[:-1])
    return {
        'model_id': model_id,
        'task': task,
    }


def get_hugging_face_pipeline_task(model_name: str):
    """
    Get the pipeline task for a Hugging Face model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Task string or None if not found
    """
    hf_api = HfApi()
    try:
        model_info = hf_api.model_info(
            model_name, token=HUGGING_FACE_HUB_TOKEN)
        task = model_info.pipeline_tag
        return task
    except Exception as e:
        logger.error(f"Error getting task for model {model_name}: {str(e)}")
        print_error("Model not found, please try another.")
        return None


def get_model_and_task(endpoint_or_model_name: str, config: Optional[Tuple[Deployment, Model]] = None) -> Dict[str, str]:
    """
    Get model ID and task
    
    Args:
        endpoint_or_model_name: Name of the endpoint or model
        config: Optional tuple of (deployment, model)
        
    Returns:
        Dictionary with model_id and task
    """
    if config is not None:
        _, model = config
        return {
            'model_id': model.id,
            'task': model.task
        }

    if (is_sagemaker_model(endpoint_or_model_name)):
        return get_sagemaker_model_and_task(endpoint_or_model_name)
    else:
        model_id = get_model_name_from_hugging_face_endpoint(
            endpoint_or_model_name)
        task = get_hugging_face_pipeline_task(model_id)
        return {
            'model_id': model_id,
            'task': task
        }


def get_model_name_from_hugging_face_endpoint(endpoint_name: str):
    """
    Get model name from Hugging Face endpoint name
    
    Args:
        endpoint_name: Name of the endpoint
        
    Returns:
        Model name
    """
    try:
        endpoint_name = endpoint_name.removeprefix("custom-")
        endpoint_name = endpoint_name.replace("--", "/")
        
        # Check if we have a valid format with author/model
        if "/" not in endpoint_name:
            # Fallback for invalid format: assume it's directly the model name
            logger.warning(f"Invalid endpoint name format: {endpoint_name}")
            return endpoint_name
            
        author, rest = endpoint_name.split("/")

        # remove datetime
        split = rest.split('-')
        fuzzy_model_name = '-'.join(split[:-1])

        # get first token
        search_term = fuzzy_model_name.split('-')[0]

        hf_api = HfApi()
        results = hf_api.list_models(search=search_term, author=author)

        # find results that closest match our fuzzy model name
        results_to_diff = {}
        for result in results:
            results_to_diff[result.id] = SequenceMatcher(
                None, result.id, f"{author}/{fuzzy_model_name}").ratio()

        if not results_to_diff:
            logger.warning(f"No models found for {author}/{fuzzy_model_name}")
            return f"{author}/{fuzzy_model_name}"
            
        return max(results_to_diff, key=results_to_diff.get)
    except Exception as e:
        logger.error(f"Error parsing endpoint name {endpoint_name}: {str(e)}")
        # Fallback to a reasonable default
        return endpoint_name


def get_text_generation_hyperpameters(config: Optional[Tuple[Deployment, Model]], query: Query = None):
    """
    Get text generation hyperparameters
    
    Args:
        config: Optional tuple of (deployment, model)
        query: Optional query with parameters
        
    Returns:
        Dictionary of hyperparameters
    """
    if query is not None and query.parameters is not None:
        return query.parameters.model_dump()

    if config is not None and config[1].predict is not None:
        return config[1].predict

    # Defaults
    return {
        "max_new_tokens": 250,
        "top_p": 0.9,
        "temperature": 0.9,
    }