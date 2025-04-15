# model-service/src/sagemaker/create_model.py
import json
import logging
import os
from dotenv import dotenv_values
from rich.table import Table
from sagemaker import image_uris, model_uris, script_uris
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.s3 import S3Uploader
from src import session
from src.config import write_config
from src.schemas.model import Model, ModelSource
from src.schemas.deployment import Deployment
from src.session import get_sagemaker_session
from src.console import console
from src.utils.aws_utils import construct_s3_uri, is_s3_uri
from src.utils.rich_utils import print_error, print_success
from src.utils.model_utils import get_unique_endpoint_name, get_model_and_task
from src.huggingface import HuggingFaceTask
from src.huggingface.hf_hub_api import get_hf_task

# Configure logging
logger = logging.getLogger(__name__)

HUGGING_FACE_HUB_TOKEN = dotenv_values(".env").get("HUGGING_FACE_HUB_KEY")
SAGEMAKER_ROLE = dotenv_values(".env").get("SAGEMAKER_ROLE")


# TODO: Consolidate
def deploy_model(deployment: Deployment, model: Model):
    match model.source:
        case ModelSource.HuggingFace:
            return deploy_huggingface_model(deployment, model)
        case ModelSource.Sagemaker:
            return create_and_deploy_jumpstart_model(deployment, model)
        case ModelSource.Custom:
            return deploy_custom_huggingface_model(deployment, model)


def deploy_huggingface_model(deployment: Deployment, model: Model):
    # Get region from environment or session
    region_name = os.environ.get("AWS_REGION_NAME") or session.region_name or "us-east-1"
    # Set it in the environment
    os.environ["AWS_REGION_NAME"] = region_name
    os.environ["AWS_DEFAULT_REGION"] = region_name
    
    logger.info(f"Using AWS region: {region_name} for HuggingFace model deployment")
    
    task = get_hf_task(model)
    model.task = task
    env = {
        'HF_MODEL_ID': model.id,
        'HF_TASK': task,
    }

    if HUGGING_FACE_HUB_TOKEN is not None:
        env['HUGGING_FACE_HUB_TOKEN'] = HUGGING_FACE_HUB_TOKEN

    image_uri = None
    if deployment.num_gpus:
        env['SM_NUM_GPUS'] = json.dumps(deployment.num_gpus)

    if deployment.quantization:
        env['HF_MODEL_QUANTIZE'] = deployment.quantization

    if task == HuggingFaceTask.TextGeneration:
        # use TGI image if llm.
        image_uri = get_huggingface_llm_image_uri(
            "huggingface",
            version="1.4.2"
        )

    # Clean up the SAGEMAKER_ROLE value
    role = SAGEMAKER_ROLE
    if role:
        # Strip quotes and whitespace
        role = role.strip().strip('"\'')
        logger.info(f"Using IAM role: {role}")

    huggingface_model = HuggingFaceModel(
        env=env,
        role=role,  # Use the cleaned role
        transformers_version="4.37",
        pytorch_version="2.1",
        py_version="py310",
        image_uri=image_uri
    )

    endpoint_name = get_unique_endpoint_name(
        model.id, deployment.endpoint_name)

    deployment.endpoint_name = endpoint_name

    logger.info(
        f"Deploying model to AWS. This may take up to 10 minutes for very large models. See full logs at: "
        f"https://{region_name}.console.aws.amazon.com/cloudwatch/home#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252F{endpoint_name}"
    )

    console.log(
        "Deploying model to AWS. [magenta]This may take up to 10 minutes for very large models.[/magenta] See full logs here:")
    console.print(
        f"https://{region_name}.console.aws.amazon.com/cloudwatch/home#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252F{endpoint_name}")

    with console.status("[bold green]Deploying model...") as status:
        table = Table(show_header=False, header_style="magenta")
        table.add_column("Resource", style="dim")
        table.add_column("Value", style="blue")
        table.add_row("model", model.id)
        table.add_row("EC2 instance type", deployment.instance_type)
        table.add_row("Number of instances", str(
            deployment.instance_count))
        table.add_row("task", task)
        console.print(table)

        try:
            predictor = huggingface_model.deploy(
                initial_instance_count=deployment.instance_count,
                instance_type=deployment.instance_type,
                endpoint_name=endpoint_name,
            )
        except Exception as e:
            logger.error(f"Failed to deploy model: {str(e)}")
            console.print_exception()
            raise

    print_success(
        f"{model.id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")

    write_config(deployment, model)
    return predictor


def deploy_custom_huggingface_model(deployment: Deployment, model: Model):
    # Get region from environment or session
    region_name = os.environ.get("AWS_REGION_NAME") or session.region_name or "us-east-1"
    # Set it in the environment
    os.environ["AWS_REGION_NAME"] = region_name
    os.environ["AWS_DEFAULT_REGION"] = region_name
    
    logger.info(f"Using AWS region: {region_name} for custom HuggingFace model deployment")
    
    if model.location is None:
        print_error("Missing model source location.")
        raise ValueError("Missing model source location")

    s3_path = model.location
    if not is_s3_uri(model.location):
        # Local file. Upload to s3 before deploying
        bucket = get_sagemaker_session().default_bucket()
        s3_path = construct_s3_uri(bucket, f"models/{model.id}")
        with console.status(f"[bold green]Uploading custom {model.id} model to S3 at {s3_path}...") as status:
            try:
                s3_path = S3Uploader.upload(
                    model.location, s3_path)
            except Exception as e:
                logger.error(f"Model failed to upload to S3: {str(e)}")
                print_error("[red] Model failed to upload to S3")
                raise

    endpoint_name = get_unique_endpoint_name(
        model.id, deployment.endpoint_name)

    deployment.endpoint_name = endpoint_name
    model.task = get_model_and_task(model.id)['task']

    console.log(
        "Deploying model to AWS. [magenta]This may take up to 10 minutes for very large models.[/magenta] See full logs here:")
    console.print(
        f"https://{region_name}.console.aws.amazon.com/cloudwatch/home#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252F{endpoint_name}")

    # Clean up the SAGEMAKER_ROLE value
    role = SAGEMAKER_ROLE
    if role:
        # Strip quotes and whitespace
        role = role.strip().strip('"\'')
        logger.info(f"Using IAM role: {role}")

    # create Hugging Face Model Class
    huggingface_model = HuggingFaceModel(
        # path to your trained sagemaker model
        model_data=s3_path,
        role=role,  # Use the cleaned role
        transformers_version="4.37",
        pytorch_version="2.1",
        py_version="py310",
    )

    with console.status("[bold green]Deploying model...") as status:
        table = Table(show_header=False, header_style="magenta")
        table.add_column("Resource", style="dim")
        table.add_column("Value", style="blue")
        table.add_row("S3 Path", s3_path)
        table.add_row("EC2 instance type", deployment.instance_type)
        table.add_row("Number of instances", str(
            deployment.instance_count))
        console.print(table)

        try:
            predictor = huggingface_model.deploy(
                initial_instance_count=deployment.instance_count,
                instance_type=deployment.instance_type,
                endpoint_name=endpoint_name
            )
        except Exception as e:
            logger.error(f"Failed to deploy model: {str(e)}")
            console.print_exception()
            raise

    print_success(
        f"Custom {model.id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")

    write_config(deployment, model)
    return predictor


def create_and_deploy_jumpstart_model(deployment: Deployment, model: Model):
    # Get region from environment or session
    region_name = os.environ.get("AWS_REGION_NAME") or session.region_name or "us-east-1"
    # Set it in the environment
    os.environ["AWS_REGION_NAME"] = region_name
    os.environ["AWS_DEFAULT_REGION"] = region_name
    
    logger.info(f"Using AWS region: {region_name} for JumpStart model deployment")
    
    endpoint_name = get_unique_endpoint_name(
        model.id, deployment.endpoint_name)
    deployment.endpoint_name = endpoint_name
    model.task = get_model_and_task(model.id)['task']

    console.log(
        "Deploying model to AWS. [magenta]This may take up to 10 minutes for very large models.[/magenta] See full logs here:")

    console.print(
        f"https://{region_name}.console.aws.amazon.com/cloudwatch/home#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252F{endpoint_name}")

    with console.status("[bold green]Deploying model...") as status:
        table = Table(show_header=False, header_style="magenta")
        table.add_column("Resource", style="dim")
        table.add_column("Value", style="blue")
        table.add_row("model", model.id)
        table.add_row("EC2 instance type", deployment.instance_type)
        table.add_row("Number of instances", str(
            deployment.instance_count))
        console.print(table)

        # Clean up the SAGEMAKER_ROLE value
        role = SAGEMAKER_ROLE
        if role:
            # Strip quotes and whitespace
            role = role.strip().strip('"\'')
            logger.info(f"Using IAM role: {role}")

        jumpstart_model = JumpStartModel(
            model_id=model.id, instance_type=deployment.instance_type, role=role)  # Use the cleaned role

        # Attempt to deploy to AWS
        try:
            predictor = jumpstart_model.deploy(
                initial_instance_count=deployment.instance_count,
                instance_type=deployment.instance_type,
                endpoint_name=endpoint_name,
                accept_eula=True
            )
        except Exception as e:
            logger.error(f"Failed to deploy model: {str(e)}")
            console.print_exception()
            raise

    write_config(deployment, model)
    print_success(
        f"{model.id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")

    return predictor