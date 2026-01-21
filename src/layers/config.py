from typing import Optional, Union, List, Callable, Any
from pydantic import BaseModel




class LayerConfig(BaseModel):
    layer_class: Optional[Callable]=None
    mode: Optional[Union[str,None]]
    layer_name: Optional[str] = None
    info_path: Optional[str] = None
    rank: Optional[int] = None
    weight_transpose: Optional[bool] = None
    # key: Optional[str] = None
    # cov_dict_path: Optional[str] = None
    covariance_matrix: Optional[Any] = None
