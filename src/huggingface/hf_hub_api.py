# model-service/src/huggingface/hf_hub_api.py

import logging
from dotenv import dotenv_values
from src.console import console
from src.huggingface import hf_api, HuggingFaceTask
from src.schemas.model import Model
from src.utils.rich_utils import print_error

logger = logging.getLogger(__name__)

HUGGING_FACE_HUB_TOKEN = dotenv_values(".env").get("HUGGING_FACE_HUB_KEY")

def get_hf_task(model: Model):
    """
    Get the task for a Hugging Face model with graceful fallback
    
    Args:
        model: Model to get task for
        
    Returns:
        Task string or default task if not found
    """
    try:
        # Try to get model info with token if available
        model_info = hf_api.model_info(model.id, token=HUGGING_FACE_HUB_TOKEN)
        task = model_info.pipeline_tag
        if model_info.transformers_info is not None and model_info.transformers_info.pipeline_tag is not None:
            task = model_info.transformers_info.pipeline_tag
        
        logger.info(f"Found task {task} for model {model.id}")
        return task
    except Exception as e:
        logger.warning(f"Could not determine task for model {model.id}: {str(e)}")
        
        # Guess task from model name for common models
        if "bert" in model.id.lower():
            logger.info(f"Fallback: Using fill-mask task for BERT model {model.id}")
            return HuggingFaceTask.FillMask
        
        logger.info(f"Fallback: Using text-generation as default task for {model.id}")
        return HuggingFaceTask.TextGeneration