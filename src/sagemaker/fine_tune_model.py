# model-service/src/sagemaker/fine_tune_model.py
import logging
import os
import sagemaker
from botocore.exceptions import ClientError
from datasets import load_dataset
from rich import print
from rich.table import Table
from sagemaker.jumpstart.estimator import JumpStartEstimator
from src.console import console
from src.schemas.model import Model, ModelSource
from src.schemas.training import Training
from src.session import get_sagemaker_session
from src.utils.aws_utils import is_s3_uri
from src.utils.rich_utils import print_success, print_error
from transformers import AutoTokenizer

from dotenv import load_dotenv
load_dotenv()
SAGEMAKER_ROLE = os.environ.get("SAGEMAKER_ROLE")

# Configure logging
logger = logging.getLogger(__name__)

def prep_hf_data(s3_bucket: str, dataset_name_or_path: str, model: Model):
    """
    Prepare Hugging Face dataset for fine-tuning
    
    Args:
        s3_bucket: S3 bucket name
        dataset_name_or_path: Dataset name or path
        model: Model to fine-tune
        
    Returns:
        Tuple of (training_path, test_path)
    """
    try:
        train_dataset, test_dataset = load_dataset(
            dataset_name_or_path, split=["train", "test"])
        tokenizer = AutoTokenizer.from_pretrained(model.id)

        def tokenize(batch):
            return tokenizer(batch["text"], padding="max_length", truncation=True)

        # tokenize train and test datasets
        train_dataset = train_dataset.map(tokenize, batched=True)
        test_dataset = test_dataset.map(tokenize, batched=True)

        # set dataset format for PyTorch
        train_dataset = train_dataset.rename_column("label", "labels")
        train_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"])
        test_dataset = test_dataset.rename_column("label", "labels")
        test_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"])

        # save train_dataset to s3
        training_input_path = f's3://{s3_bucket}/datasets/train'
        train_dataset.save_to_disk(training_input_path)

        # save test_dataset to s3
        test_input_path = f's3://{s3_bucket}/datasets/test'
        test_dataset.save_to_disk(test_input_path)

        return training_input_path, test_input_path
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise


def train_model(training: Training, model: Model, estimator):
    """
    Train a model using SageMaker
    
    Args:
        training: Training configuration
        model: Model to train
        estimator: SageMaker estimator
        
    Returns:
        Deployed predictor
    """
    # Validate training data location
    if not is_s3_uri(training.training_input_path):
        logger.error("Training data must be on S3")
        raise ValueError("Training data needs to be uploaded to s3")

    training_dataset_s3_path = training.training_input_path

    # Log training details
    table = Table(show_header=False, header_style="magenta")
    table.add_column("Resource", style="dim")
    table.add_column("Value", style="blue")
    table.add_row("model", model.id)
    table.add_row("model_version", model.version)
    table.add_row("base_model_uri", estimator.model_uri)
    table.add_row("image_uri", estimator.image_uri)
    table.add_row("EC2 instance type", training.instance_type)
    table.add_row("Number of instances", str(training.instance_count))
    console.print(table)

    # Start training job
    estimator.fit({"training": training_dataset_s3_path})

    # Deploy the trained model
    predictor = estimator.deploy(
        initial_instance_count=training.instance_count, 
        instance_type=training.instance_type
    )

    print_success(
        f"Trained model {model.id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")
    
    return predictor


def fine_tune_model(training: Training, model: Model):
    """
    Fine-tune a model
    
    Args:
        training: Training configuration
        model: Model to fine-tune
        
    Returns:
        True if successful, False otherwise
    """
    estimator = None
    
    try:
        # Create estimator based on model source
        match model.source:
            case ModelSource.Sagemaker:
                hyperparameters = get_hyperparameters_for_model(training, model)
                estimator = JumpStartEstimator(
                    model_id=model.id,
                    model_version=model.version,
                    instance_type=training.instance_type,
                    instance_count=training.instance_count,
                    output_path=training.output_path,
                    environment={"accept_eula": "true"},
                    role=SAGEMAKER_ROLE,
                    sagemaker_session=get_sagemaker_session(),
                    hyperparameters=hyperparameters
                )
            case ModelSource.HuggingFace:
                logger.error("HuggingFace fine-tuning not yet implemented")
                raise NotImplementedError("HuggingFace fine-tuning not yet implemented")
            case ModelSource.Custom:
                logger.error("Custom model fine-tuning not yet implemented")
                raise NotImplementedError("Custom model fine-tuning not yet implemented")

        # Start training job
        print_success("Enqueuing training job")
        train_model(training, model, estimator)
        return True
    except ClientError as e:
        logger.error(f"Training job enqueue failed: {str(e)}")
        print_error("Training job enqueue fail")
        return False
    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        print_error(f"Fine-tuning failed: {str(e)}")
        return False


def get_hyperparameters_for_model(training: Training, model: Model):
    """
    Get hyperparameters for a model
    
    Args:
        training: Training configuration
        model: Model to get hyperparameters for
        
    Returns:
        Dictionary of hyperparameters
    """
    try:
        hyperparameters = sagemaker.hyperparameters.retrieve_default(
            model_id=model.id, model_version=model.version)

        if training.hyperparameters is not None:
            hyperparameters.update(
                (k, v) for k, v in training.hyperparameters.model_dump().items() if v is not None)
        
        return hyperparameters
    except Exception as e:
        logger.error(f"Failed to get hyperparameters: {str(e)}")
        raise