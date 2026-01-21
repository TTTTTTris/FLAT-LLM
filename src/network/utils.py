
import torch
import os
import pickle
from src.layers.config import LayerConfig
from src.network.config import NetConfig
from src import layers
import torch.nn as nn



def save_model_params(the_model, net_config_dict, dir):
    model_params_dict={}
    target_layers=net_config_dict['target_layers']
    for name, module in the_model.named_modules():
        if name in target_layers:
            # print(name)
            index = target_layers.index(name)
            key=net_config_dict['layer_configs'][index]['layer_name']
            model_params_dict[key] = module.get_params_dict()
    
    tm_params_path=os.path.join(dir,'tm_params.pth')
    torch.save(model_params_dict, tm_params_path)


def process_net_config(net_config_dict):
    processed_net_config_dict = {
                'target_layers': net_config_dict['target_layers'],
                'target_types': net_config_dict['target_types'],
                # 'layer_configs': layer_configs_list
            }
    layer_config_list = []
    for conf in net_config_dict['layer_configs']:
        conf['layer_class'] = getattr(layers, conf['layer_class_type']) 
        layer_config_list.append(LayerConfig(**conf))
    processed_net_config_dict['layer_configs'] = layer_config_list
    net_config=NetConfig(**processed_net_config_dict)
    return net_config

