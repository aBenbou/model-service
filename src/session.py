# model-service/src/session.py
import boto3
import sagemaker
import os
import logging

logger = logging.getLogger(__name__)

# Set region in environment variables
region_name = os.environ.get("AWS_REGION_NAME") or "us-east-1"
os.environ["AWS_REGION_NAME"] = region_name
os.environ["AWS_DEFAULT_REGION"] = region_name

# Create boto3 session with explicit region
session = boto3.session.Session(region_name=region_name)

# Initialize SageMaker session only when needed
sagemaker_session = None

def get_sagemaker_session():
    """
    Get or create SageMaker session with proper error handling.
    
    Returns:
        sagemaker.session.Session or None if not configured
    """
    global sagemaker_session
    
    if sagemaker_session is not None:
        return sagemaker_session
        
    try:
        # Ensure region is set
        region = os.environ.get("AWS_REGION_NAME") or "us-east-1" 
        
        logger.info(f"Initializing SageMaker session with region: {region}")
        
        # Create session with explicitly set region
        sagemaker_session = sagemaker.session.Session(
            boto_session=boto3.session.Session(region_name=region)
        )
        
        return sagemaker_session
    except Exception as e:
        logger.error(f"Failed to initialize SageMaker session: {str(e)}")
        logger.warning("Running with limited functionality. AWS operations will not work.")
        return None