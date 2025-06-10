import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 输入: 3通道 32x32 (CIFAR10)
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),  # 输出: 48x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 48x16x16
            
            nn.Conv2d(48, 128, kernel_size=3, padding=1),  # 输出: 128x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 128x8x8
            
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # 输出: 192x8x8
            nn.ReLU(inplace=True),
            
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # 输出: 192x8x8
            nn.ReLU(inplace=True),
            
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # 输出: 128x8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出: 128x4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10)  # CIFAR10有10个类别
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # 测试网络结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建测试输入 (batch_size=1, channels=3, height=32, width=32)
    test_input = torch.randn(1, 3, 32, 32).to(device)
    
    # 初始化模型
    model = AlexNet().to(device)
    
    # 前向传播测试
    output = model(test_input)
    print(f"输入尺寸: {test_input.shape}")
    print(f"输出尺寸: {output.shape}")  # 应为 torch.Size([1, 10])
