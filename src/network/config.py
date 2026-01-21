from typing import Optional, Union, List, Callable
from pydantic import BaseModel, ValidationError, field_validator, model_validator

class NetConfig(BaseModel):
    target_layers: List[str] 
    target_types: List[str] 
    layer_configs: List[BaseModel] 
    class Config:
        arbitrary_types_allowed = True
