import torch
import torch.nn as nn
from torchvision.models import (
    # ResNet variants
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    # ResNeXt variants
    resnext50_32x4d, ResNeXt50_32X4D_Weights,
    resnext101_32x8d, ResNeXt101_32X8D_Weights,
    resnext101_64x4d, ResNeXt101_64X4D_Weights,
    # Swin variants
    swin_t, Swin_T_Weights,
    swin_s, Swin_S_Weights,
    swin_b, Swin_B_Weights,
    # MobileNet variants
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
)

class BackboneFactory(nn.Module):
    """
    Backbone model factory that extracts features with 8x downsampling from input image
    Supported models:
        - ResNet (18, 34, 50, 101)
        - ResNeXt (50-32x4d, 101-32x8d, 101-64x4d)
        - Swin (tiny, small, base)
        - MobileNetV3 (small, large)
    """
    SUPPORTED_MODELS = {
        # ResNet family
        'resnet18': (resnet18, ResNet18_Weights),
        'resnet34': (resnet34, ResNet34_Weights),
        'resnet50': (resnet50, ResNet50_Weights),
        'resnet101': (resnet101, ResNet101_Weights),
        # ResNeXt family
        'resnext50': (resnext50_32x4d, ResNeXt50_32X4D_Weights),
        'resnext101_32x8d': (resnext101_32x8d, ResNeXt101_32X8D_Weights),
        'resnext101_64x4d': (resnext101_64x4d, ResNeXt101_64X4D_Weights),
        # Swin family
        'swin_t': (swin_t, Swin_T_Weights),
        'swin_s': (swin_s, Swin_S_Weights),
        'swin_b': (swin_b, Swin_B_Weights),
        # MobileNet family
        'mobilenet_v3_small': (mobilenet_v3_small, MobileNet_V3_Small_Weights),
        'mobilenet_v3_large': (mobilenet_v3_large, MobileNet_V3_Large_Weights),
    }

    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        self.model_name = model_name.lower()
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. "
                           f"Supported models are: {list(self.SUPPORTED_MODELS.keys())}")
        self.backbone = self._create_backbone(pretrained)
        self.output_channels = self._get_output_channels()
        
    def _create_backbone(self, pretrained):
        """Creates backbone model up to the layer that gives 8x downsampling"""
        model_fn, weights_enum = self.SUPPORTED_MODELS[self.model_name]
        weights = weights_enum.DEFAULT if pretrained else None
        model = model_fn(weights=weights)
        
        if self.model_name.startswith('resnet') or self.model_name.startswith('resnext'):
            return nn.Sequential(
                model.conv1,      # /2
                model.bn1,
                model.relu,
                model.maxpool,    # /2
                model.layer1,     
                model.layer2,     # /2
                model.layer3,     # /2
            )
            
        elif self.model_name.startswith('swin'):
            return nn.Sequential(
                model.features[:2],    # patch_embed + layers[0], /4
                model.features[2],     # layers[1], /2
            )
            
        elif self.model_name.startswith('mobilenet'):
            if 'small' in self.model_name:
                # MobileNetV3-Small has different structure
                return nn.Sequential(
                    model.features[0],     # /2
                    *model.features[1:8]   # /4
                )
            else:
                # MobileNetV3-Large
                return nn.Sequential(
                    model.features[0],     # /2
                    *model.features[1:7]   # /4
                )
    
    def _get_output_channels(self):
        """
        Returns the number of output channels for the backbone
        Uses a dummy forward pass to determine the output shape
        """
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.backbone(dummy)
            return out.shape[1]
    
    def forward(self, x):
        return self.backbone(x)

class BackboneInfo:
    """
    Utility class providing information about supported backbone models
    """
    @staticmethod
    def get_model_info():
        """
        Returns a dictionary containing detailed information about each model variant:
        - Available variants for each model family
        - Number of parameters
        - Output channels
        """
        info = {
            "ResNet Family": {
                "variants": ["resnet18", "resnet34", "resnet50", "resnet101"],
                "characteristics": {
                    "resnet18": {"params": "11M", "output_channels": 256},
                    "resnet34": {"params": "21M", "output_channels": 256},
                    "resnet50": {"params": "23M", "output_channels": 1024},
                    "resnet101": {"params": "42M", "output_channels": 1024},
                }
            },
            "ResNeXt Family": {
                "variants": ["resnext50", "resnext101_32x8d", "resnext101_64x4d"],
                "characteristics": {
                    "resnext50": {"params": "23M", "output_channels": 1024},
                    "resnext101_32x8d": {"params": "86M", "output_channels": 1024},
                    "resnext101_64x4d": {"params": "83M", "output_channels": 1024},
                }
            },
            "Swin Family": {
                "variants": ["swin_t", "swin_s", "swin_b"],
                "characteristics": {
                    "swin_t": {"params": "28M", "output_channels": 384},
                    "swin_s": {"params": "50M", "output_channels": 768},
                    "swin_b": {"params": "88M", "output_channels": 1024},
                }
            },
            "MobileNet Family": {
                "variants": ["mobilenet_v3_small", "mobilenet_v3_large"],
                "characteristics": {
                    "mobilenet_v3_small": {"params": "2.5M", "output_channels": 96},
                    "mobilenet_v3_large": {"params": "5.4M", "output_channels": 160},
                }
            }
        }
        return info

# Usage example
if __name__ == "__main__":
    # Print model information
    info = BackboneInfo.get_model_info()
    print("Available Backbone Models:")
    for family, details in info.items():
        print(f"\n{family}:")
        for variant in details["variants"]:
            chars = details["characteristics"][variant]
            print(f"  - {variant}:")
            print(f"    Parameters: {chars['params']}")
            print(f"    Output Channels: {chars['output_channels']}")
    
    # Test models
    test_models = [
        'resnet50',
        'resnext50',
        'swin_t',
        'mobilenet_v3_large'
    ]
    
    input_tensor = torch.randn(1, 3, 224, 224)
    
    for model_name in test_models:
        print(f"\nTesting {model_name}...")
        model = BackboneFactory(model_name=model_name, pretrained=True)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output channels: {model.output_channels}")
        print(f"Downsample ratio: {input_tensor.shape[-1] // output.shape[-1]}x")