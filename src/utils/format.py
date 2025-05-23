# model-service/src/utils/format.py
from typing import Dict


def to_camel_case(snake_str: str) -> str:
    """Convert snake_case to CamelCase"""
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def to_lower_camel_case(snake_str: str) -> str:
    """Convert snake_case to lowerCamelCase"""
    # We capitalize the first letter of each component except the first one
    # with the 'capitalize' method and join them together.
    camel_string = to_camel_case(snake_str)
    return lower_first(camel_string)


def lower_first(string):
    """Make the first character lowercase"""
    return string[0].lower() + string[1:]


def format_sagemaker_endpoint(output: Dict[str, str]) -> Dict[str, str]:
    """Format SageMaker endpoint info to lowerCamelCase keys"""
    return {lower_first(key): val for key, val in output.items()}


def format_python_dict(output: Dict[str, str]):
    """Format Python dict keys to lowerCamelCase"""
    return {to_lower_camel_case(key): val for key, val in output.items()}