import torch
import torch.nn as nn
import timm

def build_swinv2(model_name="swinv2_tiny_window16_256", num_labels=3, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_labels) 
    return model