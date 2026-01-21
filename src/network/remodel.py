
import torch.nn as nn
import copy
from . import utils
from tqdm import tqdm

def layer_type_mapping(layer_type_str):
    layer_type_mapping = {
        "linear": nn.Linear,
        "embedding": nn.Embedding
        # Add more mappings as needed
    }
    return layer_type_mapping.get(layer_type_str.lower())


def get_parent_module(input_model,layer_name, layer_class):
    found_layer=False
    for name, module in input_model.named_modules():
        if name == layer_name:
            found_layer=True
            if isinstance(module, layer_class):
                parent_name, attr_name = name.rsplit('.', 1) if '.' in name else (None, name)
                if parent_name is None:
                    attr_name=name
                    parent_module=input_model
                else:
                    parent_module = dict(input_model.named_modules())[parent_name]
            else:
                raise TypeError(f"Layer {layer_name} type mismatch. Expected {layer_class}, got {type(module)}.")
    if not found_layer:
        raise ValueError(f"Layer {layer_name} not found in the model.")
    return parent_module, attr_name


def substitute_layer(input_model, layer_name, layer_type, substitute_layer): 
    layer_class=layer_type_mapping(layer_type)
    if not layer_class:
        raise ValueError(f"Layer type '{layer_type}' is not supported.")
    parent_module,target_name=get_parent_module(input_model,layer_name,layer_class)
    setattr(parent_module, target_name, substitute_layer)
    return input_model

def setup_layer_from_model(input_model, layer_name, layer_type, new_layer_config, params_dict=None):
    target_layer_class=layer_type_mapping(layer_type)
    if not target_layer_class:
        raise ValueError(f"Layer type '{layer_type}' is not supported.")
    parent_module, target_name=get_parent_module(input_model,layer_name, target_layer_class)
    target_layer=getattr(parent_module, target_name)
    if params_dict!=None:
        new_layer=new_layer_config.layer_class.from_file(params_dict, new_layer_config)
    else:
        new_layer=new_layer_config.layer_class.from_teacher(target_layer, new_layer_config)
    device = next(input_model.parameters()).device
    new_layer = new_layer.to(device)
    return new_layer


def swap_layers(the_model, net_config_dict, model_params_dict=None):
    net_config=utils.process_net_config(net_config_dict)
    target_layers=net_config.target_layers
    target_types=net_config.target_types
    new_layers_config=net_config.layer_configs

    for index, target_name in tqdm(enumerate(target_layers), total=len(target_layers), desc="Remodel"):
        if model_params_dict!=None:
            params_dict=model_params_dict
            new_layer_instance=setup_layer_from_model(the_model,target_name,target_types[index],new_layers_config[index], params_dict)
        else:
            new_layer_instance=setup_layer_from_model(the_model,target_name,target_types[index],new_layers_config[index])
        the_model=substitute_layer(the_model,target_name,target_types[index],new_layer_instance)
    
    return the_model
