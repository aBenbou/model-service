# model-service/src/sagemaker/resources.py
import os
import boto3
import logging
from functools import lru_cache
from InquirerPy import inquirer
from src.console import console
from src.sagemaker import EC2Instance
from src.config import get_config_for_endpoint
from src.utils.format import format_sagemaker_endpoint, format_python_dict
from src.session import session
from typing import List, Tuple, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

def list_sagemaker_endpoints(filter_str: str = None) -> List[Dict]:
    """
    List active SageMaker endpoints with instance information
    
    Args:
        filter_str: Optional filter string to match endpoint names
        
    Returns:
        List of endpoint dictionaries with instance type information
    """
    try:
        # Use the session's region, or get it from environment variables
        region_name = session.region_name or os.environ.get('AWS_REGION_NAME') or 'us-east-1'
        logger.info(f"Using AWS region: {region_name}")
        
        # Create client with explicit region
        sagemaker_client = boto3.client('sagemaker', region_name=region_name)

        endpoints = sagemaker_client.list_endpoints()['Endpoints']
        
        if filter_str is not None:
            endpoints = list(filter(lambda x: filter_str == x['EndpointName'], endpoints))

        # Add instance type information to each endpoint
        for endpoint in endpoints:
            try:
                endpoint_config = sagemaker_client.describe_endpoint_config(
                    EndpointConfigName=endpoint['EndpointName'])['ProductionVariants'][0]
                endpoint['InstanceType'] = endpoint_config['InstanceType']
            except Exception as e:
                logger.warning(f"Could not get instance type for endpoint {endpoint['EndpointName']}: {str(e)}")
                endpoint['InstanceType'] = "unknown"
                
        return endpoints
    except Exception as e:
        logger.error(f"Error listing SageMaker endpoints: {str(e)}")
        raise  # Re-raise the exception to propagate it up


def get_sagemaker_endpoint(endpoint_name: str) -> Optional[Dict[str, Optional[Dict]]]:
    """
    Get detailed information about a specific endpoint
    
    Args:
        endpoint_name: Name of the endpoint to retrieve
        
    Returns:
        Dictionary with deployment and model information
    """
    endpoints = list_sagemaker_endpoints(endpoint_name)
    if not endpoints:
        return None

    endpoint = format_sagemaker_endpoint(endpoints[0])

    config = get_config_for_endpoint(endpoint_name)
    if config is None:
        return {'deployment': endpoint, 'model': None}

    deployment = format_python_dict(config.deployment.model_dump())
    formatted_models = []
    for model in config.models:
        model = format_python_dict(model.model_dump())
        formatted_models.append(model)

    # Merge the endpoint dict with our config
    deployment = {**endpoint, **deployment}

    return {
        'deployment': deployment,
        'models': formatted_models,
    }


@lru_cache
def list_service_quotas() -> List[Tuple[str, int]]:
    """
    Gets a list of EC2 instances for inference with their quotas
    
    Returns:
        List of tuples containing (instance_name, quota_value)
    """
    # Use the session's region, or get it from environment variables
    region_name = session.region_name or os.environ.get('AWS_REGION_NAME') or 'us-east-1'
    logger.info(f"Using AWS region: {region_name}")
    
    client = boto3.client('service-quotas', region_name=region_name)
    quotas = []
    
    try:
        response = client.list_service_quotas(
            ServiceCode="sagemaker",
            MaxResults=100,
        )
        next_token = response.get('NextToken')
        quotas = response['Quotas']
        
        while next_token is not None:
            response = client.list_service_quotas(
                ServiceCode="sagemaker",
                NextToken=next_token,
            )
            quotas.extend(response['Quotas'])
            next_token = response.get('NextToken')
    except Exception as e:
        logger.error(f"Error getting service quotas: {str(e)}")
        console.print(
            "[red]User does not have access to Service Quotas. Grant access via IAM to get the list of available instances")
        return []

    # Filter for inference instance quotas
    available_instances = list(filter(lambda x: 'endpoint usage' in x[0] and x[1] > 0, [
                            (quota['QuotaName'], quota['Value']) for quota in quotas]))

    # Clean up quota names
    available_instances = [(instance[0].split(" ")[0], instance[1])
                        for instance in available_instances]
    return available_instances


def list_service_quotas_async(instances=[]):
    """Wrapper to allow access to list in threading"""
    quota_instances = list_service_quotas()
    if isinstance(instances, list):
        instances.extend(quota_instances)
    return quota_instances


def select_instance(available_instances=None):
    """
    Interactive selection of a SageMaker instance type
    
    Args:
        available_instances: Optional list of available instances
        
    Returns:
        Selected instance type
    """
    choices = [instance[0] for instance in available_instances] if available_instances else [
        instance for instance in EC2Instance]
    
    instance = inquirer.fuzzy(
        message="Choose an instance size (note: ml.m5.xlarge available by default; you must request quota from AWS to use other instance types):",
        choices=choices,
        default="ml.m5."
    ).execute()
    
    if instance is None:
        return EC2Instance.SMALL  # Default to small if nothing selected
        
    return instance