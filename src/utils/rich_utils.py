# model-service/src/utils/rich_utils.py
from rich import print


def print_model(text: str):
    """Print model information in blue"""
    print(f"[blue] {text}")


def print_error(text: str):
    """Print error in red"""
    print(f"[red] {text}")


def print_success(text: str):
    """Print success message in green"""
    print(f"[green] {text}")