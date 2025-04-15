# model-service/src/sagemaker/query_endpoint.py
import boto3
import json
import logging
import inquirer
from InquirerPy import prompt
from sagemaker.huggingface.model import HuggingFacePredictor
from src.config import ModelDeployment
from src.console import console
from src.sagemaker import SagemakerTask
from src.huggingface import HuggingFaceTask
from src.utils.model_utils import get_model_and_task, is_sagemaker_model, get_text_generation_hyperpameters
from src.utils.rich_utils import print_error
from src.schemas.deployment import Deployment
from src.schemas.model import Model
from src.schemas.query import Query
from src.session import get_sagemaker_session
from typing import Dict, Tuple, Optional
import os

# Configure logging
logger = logging.getLogger(__name__)

def make_query_request(endpoint_name: str, query: Query, config: Tuple[Deployment, Model]):
    if is_sagemaker_model(endpoint_name, config):
        return query_sagemaker_endpoint(endpoint_name, query, config)
    else:
        return query_hugging_face_endpoint(endpoint_name, query, config)


def parse_response(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    probabilities, labels, predicted_label = model_predictions[
        'probabilities'], model_predictions['labels'], model_predictions['predicted_label']
    return probabilities, labels, predicted_label


def query_hugging_face_endpoint(endpoint_name: str, user_query: Query, config: Tuple[Deployment, Model]):
    task = get_model_and_task(endpoint_name, config)['task']
    predictor = HuggingFacePredictor(endpoint_name=endpoint_name,
                                     sagemaker_session=get_sagemaker_session())

    query = user_query.query
    context = user_query.context

    input = {"inputs": query}
    if task is not None and task == HuggingFaceTask.QuestionAnswering:
        if context is None:
            questions = [{
                "type": "input", "message": "What context would you like to provide?:", "name": "context"}]
            answers = prompt(questions)
            context = answers.get('context', '')

        if not context:
            logger.error("Missing context for question-answering task")
            raise ValueError("Must provide context for question-answering")

        input = {}
        input['context'] = context
        input['question'] = query

    if task is not None and task == HuggingFaceTask.TextGeneration:
        parameters = get_text_generation_hyperpameters(config, user_query)
        input['parameters'] = parameters

    if task is not None and task == HuggingFaceTask.ZeroShotClassification:
        if context is None:
            questions = [
                inquirer.Text('labels',
                              message="What labels would you like to use? (comma separated values)?",
                              )
            ]
            answers = inquirer.prompt(questions)
            context = answers.get('labels', '')

        if not context:
            logger.error("Missing labels for zero-shot classification task")
            raise ValueError("Must provide labels for zero shot text classification")

        labels = context.split(',')
        input = json.dumps({
            "sequences": query,
            "candidate_labels": labels
        })

    try:
        logger.info(f"Querying HuggingFace endpoint {endpoint_name}")
        result = predictor.predict(input)
        return result
    except Exception as e:
        logger.error(f"Failed to query HuggingFace endpoint: {str(e)}")
        console.print_exception()
        raise


def query_sagemaker_endpoint(endpoint_name: str, user_query: Query, config: Tuple[Deployment, Model]):
    # Use the session's region, or get it from environment variables
    region_name = get_sagemaker_session().boto_session.region_name or os.environ.get('AWS_REGION_NAME') or 'us-east-1'
    logger.info(f"Using AWS region: {region_name}")
    
    client = boto3.client('runtime.sagemaker', region_name=region_name)
    task = get_model_and_task(endpoint_name, config)['task']

    if task not in [
        SagemakerTask.ExtractiveQuestionAnswering,
        SagemakerTask.TextClassification,
        SagemakerTask.SentenceSimilarity,
        SagemakerTask.SentencePairClassification,
        SagemakerTask.Summarization,
        SagemakerTask.NamedEntityRecognition,
        SagemakerTask.TextEmbedding,
        SagemakerTask.TcEmbedding,
        SagemakerTask.TextGeneration,
        SagemakerTask.TextGeneration1,
        SagemakerTask.TextGeneration2,
        SagemakerTask.Translation,
        SagemakerTask.FillMask,
        SagemakerTask.ZeroShotTextClassification
    ]:
        error_msg = f"Querying model type {task} is not yet supported"
        logger.error(error_msg)
        print_error("""
Querying this model type inside of Model Manager isn't yet supported. 
You can query it directly through the API endpoint - see here for documentation on how to do this:
https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html
                    """)
        raise ValueError(error_msg)

    # MIME content type varies per deployment
    content_type = "application/x-text"
    accept_type = "application/json;verbose"

    # Depending on the task, input needs to be formatted differently.
    # e.g. question-answering needs to have {question: , context: }
    query = user_query.query
    context = user_query.context
    input = query.encode("utf-8")
    match task:
        case SagemakerTask.ExtractiveQuestionAnswering:
            if context is None:
                questions = [
                    {
                        'type': 'input',
                        'name': 'context',
                        'message': "What context would you like to provide?",
                    }
                ]
                answers = prompt(questions)
                context = answers.get("context", '')

            if not context:
                logger.error("Missing context for question-answering task")
                raise ValueError("Must provide context for question-answering")

            content_type = "application/list-text"
            input = json.dumps([query, context]).encode("utf-8")

        case SagemakerTask.SentencePairClassification:
            if context is None:
                questions = [
                    inquirer.Text('context',
                                  message="What sentence would you like to compare against?",
                                  )
                ]
                answers = inquirer.prompt(questions)
                context = answers.get("context", '')
            if not context:
                logger.error("Missing second sentence for sentence pair classification")
                raise ValueError("Must provide a second sentence for sentence pair classification")

            content_type = "application/list-text"
            input = json.dumps([query, context]).encode("utf-8")
        case SagemakerTask.ZeroShotTextClassification:
            if context is None:
                questions = [
                    inquirer.Text('labels',
                                  message="What labels would you like to use? (comma separated values)?",
                                  )
                ]
                answers = inquirer.prompt(questions)
                context = answers.get('labels', '')

            if not context:
                logger.error("Missing labels for zero-shot classification")
                raise ValueError("must provide labels for zero shot text classification")
            labels = context.split(',')

            content_type = "application/json"
            input = json.dumps({
                "sequences": query,
                "candidate_labels": labels,
            }).encode("utf-8")
        case SagemakerTask.TextGeneration:
            parameters = get_text_generation_hyperpameters(config, user_query)
            input = json.dumps({
                "inputs": query,
                "parameters": parameters,
            }).encode("utf-8")
            content_type = "application/json"

    try:
        logger.info(f"Querying SageMaker endpoint {endpoint_name}")
        response = client.invoke_endpoint(
            EndpointName=endpoint_name, ContentType=content_type, Body=input, Accept=accept_type)
    except Exception as e:
        logger.error(f"Failed to query SageMaker endpoint: {str(e)}")
        console.print_exception()
        raise

    model_predictions = json.loads(response['Body'].read())
    return model_predictions