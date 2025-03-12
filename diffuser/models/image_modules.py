import torch
import torch.nn as nn
from transformers import AutoModel
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class AutoEncoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super(AutoEncoder, self).__init__()
        
        # 编码器
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 128x128x3 -> 64x64x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x64x32 -> 32x32x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32x64 -> 16x16x128
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16x128 -> 8x8x256
            nn.ReLU(),
            nn.Flatten(),  # 8x8x256 -> 16384
            nn.Linear(8 * 8 * 256, hidden_dim)  # 16384 -> hidden_dim
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 8 * 8 * 256),  # hidden_dim -> 16384
            nn.Unflatten(1, (256, 8, 8)),  # 16384 -> 8x8x256
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8x256 -> 16x16x128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16x128 -> 32x32x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32x64 -> 64x64x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 64x64x32 -> 128x128x3
            nn.Sigmoid()  # 将输出值限制在 [0, 1] 范围内
        )
    
    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        # 解码
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    @property
    def num_channels(self):
        "for compatibility with other models"
        return self.hidden_dim

class AutoEncoderTanh(AutoEncoder):
    def __init__(self, hidden_dim=256):
        super(AutoEncoderTanh, self).__init__(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 128x128x3 -> 64x64x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x64x32 -> 32x32x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32x64 -> 16x16x128
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16x128 -> 8x8x256
            nn.ReLU(),
            nn.Flatten(),  # 8x8x256 -> 16384
            nn.Linear(8 * 8 * 256, hidden_dim),  # 16384 -> hidden_dim
            nn.Tanh()
        )


class ResnetAutoEncoder(nn.Module):
    def __init__(self, hidden_dim=256, pretrained=True, freezed=False, name="resnet18", return_interm_layers=False):
        super(ResnetAutoEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        assert name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
        self.num_channels = self._get_num_channels(name)
        self.pretrained = pretrained
        self.freezed = freezed
        self.resnet = self._get_resnet(name)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(self.resnet, return_layers=return_layers)
    
    def encode(self, x):
        features = self.body(x)
        return features
    
    def decode(self, x):
        return self.decoder(x)

    def _get_num_channels(self, name):
        if name == "resnet18":
            return 512
        elif name == "resnet34":
            return 512
        elif name == "resnet50":
            return 2048
        elif name == "resnet101":
            return 2048
        elif name == "resnet152":
            return 2048
        
    def _get_resnet(self, name):
        kwargs = {
            "replace_stride_with_dilation": [False, False, True],
            "pretrained": self.pretrained,
            "norm_layer": FrozenBatchNorm2d,
        }
        if name == "resnet18":
            return models.resnet18(**kwargs)
        elif name == "resnet34":
            return models.resnet34(**kwargs)
        elif name == "resnet50":
            return models.resnet50(**kwargs)
        elif name == "resnet101":
            return models.resnet101(**kwargs)
        elif name == "resnet152":
            return models.resnet152(**kwargs)

