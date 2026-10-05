"""MONAI UNet with GCD layer metadata; checkpoint keys stay unchanged."""
import torch
from monai.networks.nets import UNet
from src.utils.register import MODEL


@MODEL.register_module()
class MonaiUNet(UNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layers = [name for name, module in self.named_modules()
                  if isinstance(module, torch.nn.Conv3d)][-4:-1]
        self.xai_layer_targets = {name: name for name in layers}
