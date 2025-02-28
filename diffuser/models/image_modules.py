import torch
import torch.nn as nn

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
