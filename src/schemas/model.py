# model-service/src/schemas/model.py
import yaml
from src.yaml import loader, dumper
from typing import Optional, Union, Dict, Any, List
from enum import Enum
from src.huggingface import HuggingFaceTask
from src.sagemaker import SagemakerTask
from pydantic import BaseModel


class ModelSource(str, Enum):
    HuggingFace = "huggingface"
    Sagemaker = "sagemaker"
    Custom = "custom"


class Model(BaseModel):
    id: str
    source: ModelSource
    task: Optional[str] = None
    version: Optional[str] = None
    location: Optional[str] = None
    predict: Optional[Dict[str, Any]] = None


class ModelResponse(BaseModel):
    id: str
    source: str
    task: Optional[str] = None
    version: Optional[str] = None
    endpoint_name: Optional[str] = None
    status: Optional[str] = None


class ModelList(BaseModel):
    models: List[ModelResponse]
    total: int
    page: int
    per_page: int


def model_representer(dumper: yaml.SafeDumper, model: Model) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping("!Model", {
        "id": model.id,
        "source": model.source.value,
        "task": model.task,
        "version": model.version,
        "location": model.location,
        "predict": model.predict,
    })


def model_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Model:
    return Model(**loader.construct_mapping(node))


dumper.add_representer(Model, model_representer)
loader.add_constructor(u'!Model', model_constructor)