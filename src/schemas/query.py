# model-service/src/schemas/query.py
from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class QueryParameters(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


class Query(BaseModel):
    query: str
    context: Optional[str] = None
    parameters: Optional[QueryParameters] = None


class QueryResponse(BaseModel):
    response: str
    model_id: str
    endpoint_name: str
    parameters: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None