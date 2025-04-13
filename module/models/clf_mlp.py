import torch
import torch.nn as nn

'''
class mlp(nn.Module):
    def __init__(self, num_classes=3, num_tokens = 96):
        super(mlp, self).__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.head = nn.Linear(num_tokens, num_outputs)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x -> (b, 96, 4, 4, 4, t)
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        # x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
        
'''
class mlp(nn.Module):
    def __init__(self, num_classes=3, num_tokens=192):
        super(mlp, self).__init__()
        self.head = nn.Linear(num_tokens, num_classes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 192, -1)  # 展平成 [batch_size, 192, 160]
        x = self.avgpool(x)  # 池化到 [batch_size, 192, 1]
        x = torch.flatten(x, 1)  # 展平到 [batch_size, 192]
        x = self.head(x)  # 输出 [batch_size, num_classes]
        return x

