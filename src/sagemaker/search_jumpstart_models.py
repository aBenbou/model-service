# model-service/src/sagemaker/search_jumpstart_models.py
import inquirer
import logging
from enum import StrEnum, auto
from sagemaker.jumpstart.notebook_utils import list_jumpstart_models
from src.utils.rich_utils import print_error
from src.session import get_sagemaker_session

# Configure logging
logger = logging.getLogger(__name__)

class Frameworks(StrEnum):
    huggingface = auto()
    meta = auto()
    model = auto()
    tensorflow = auto()
    pytorch = auto()
    # autogluon
    # catboost
    # lightgbm
    mxnet = auto()
    # sklearn
    # xgboost


def search_sagemaker_jumpstart_model(framework=None):
    """
    Search for models in SageMaker JumpStart
    
    Args:
        framework: Optional framework filter
        
    Returns:
        List of available models
    """
    if framework is None:
        # Interactive selection if no framework provided
        questions = [
            inquirer.List('framework',
                        message="Which framework would you like to use?",
                        choices=[framework.value for framework in Frameworks]
                        ),
        ]
        answers = inquirer.prompt(questions)
        if answers is None:
            return []
        framework = answers["framework"]
    
    filter_value = f"framework == {framework}"
    
    try:
        logger.info(f"Searching SageMaker JumpStart models with filter: {filter_value}")
        models = list_jumpstart_models(
            filter=filter_value,
            region=get_sagemaker_session().boto_session.region_name,
            sagemaker_session=get_sagemaker_session()
        )
        return models
    except Exception as e:
        logger.error(f"Error searching SageMaker models: {str(e)}")
        print_error(f"Failed to search SageMaker models: {str(e)}")
        return []