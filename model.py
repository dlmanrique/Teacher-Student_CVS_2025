import torch
import torch.nn as nn
import timm
import os
import random
import numpy as np

from swinv2 import SwinTransformerV2


seed = 5
# Environment Standardisation
random.seed(seed)                      # Set random seed
np.random.seed(seed)                   # Set NumPy seed
torch.manual_seed(seed)                # Set PyTorch seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)           # Set CUDA seed
torch.use_deterministic_algorithms(True) # Force deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # CUDA workspace config


def build_swinv2(model_name="swinv2_tiny_window16_256", num_labels=3, pretrained=True):
    
    model = timm.create_model(model_name, pretrained=pretrained)
    in_features = model.head.in_features
    model.head.fc = nn.Linear(in_features, num_labels) 
    return model


def build_teacher(ckpt_path):
    # Initialise Backbone
    model = SwinTransformerV2( img_size=384, # important to load correct IMNET pretrained weights
                                patch_size=4,
                                in_chans=3,
                                num_classes=3,
                                embed_dim=128,
                                depths=[ 2, 2, 18, 2 ],
                                num_heads=[ 4, 8, 16, 32 ],
                                window_size=24,
                                mlp_ratio=4,
                                qkv_bias=True,
                                drop_rate=0.0,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False,
                                pretrained_window_sizes=[ 12, 12, 12, 6 ])
    
    # Load imagenet weights onto the backbone
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    print(f'Loaded weigths from : {ckpt_path}')

    return model
