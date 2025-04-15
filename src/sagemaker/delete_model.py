# model-service/src/sagemaker/delete_model.py
import boto3
import logging
from rich import print
from src.utils.rich_utils import print_success
from typing import List
from src.session import session
import os

# Configure logging
logger = logging.getLogger(__name__)

def delete_sagemaker_model(endpoint_names: List[str] = None):
    # Use the session's region, or get it from environment variables
    region_name = session.region_name or os.environ.get('AWS_REGION_NAME') or 'us-east-1'
    logger.info(f"Using AWS region: {region_name}")
    
    sagemaker_client = boto3.client('sagemaker', region_name=region_name)

    if not endpoint_names or len(endpoint_names) == 0:
        logger.info("No endpoints to delete")
        print_success("No Endpoints to delete!")
        return

    # Add validation / error handling
    for endpoint in endpoint_names:
        logger.info(f"Deleting endpoint: {endpoint}")
        print(f"Deleting [blue]{endpoint}")
        try:
            sagemaker_client.delete_endpoint(EndpointName=endpoint)
        except Exception as e:
            logger.error(f"Failed to delete endpoint {endpoint}: {str(e)}")
            print(f"[red]Error deleting {endpoint}: {str(e)}")