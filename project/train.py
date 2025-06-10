# 完整的模型训练套路(以CIFAR10为例)
import time
import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from alex import alex  # 假设alex.py中定义了AlexNet模型

def main():
    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    
    # 1. 准备数据集
    transform = torchvision.transforms.ToTensor()
    train_data = torchvision.datasets.CIFAR10(
        root="../dataset_chen",
        train=True,
        transform=transform,
        download=True
    )
    test_data = torchvision.datasets.CIFAR10(
        root="../dataset_chen",
        train=False,
        transform=transform,
        download=True
    )

    # 数据集信息
    train_size, test_size = len(train_data), len(test_data)
    print(f"训练集大小: {train_size}, 测试集大小: {test_size}")

    # 2. 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # 3. 初始化模型、损失函数和优化器
    model = alex()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 4. 训练配置
    num_epochs = 10
    train_steps = 0
    test_steps = 0
    writer = SummaryWriter("../logs_train")
    start_time = time.time()

    # 5. 训练循环
    for epoch in range(num_epochs):
        print(f"\n----- 第 {epoch+1}/{num_epochs} 轮训练开始 -----")
        
        # 训练阶段
        model.train()
        for images, labels in train_loader:
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录训练信息
            train_steps += 1
            if train_steps % 500 == 0:
                print(f"训练步数: {train_steps}, Loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), train_steps)

        # 测试阶段
        model.eval()
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                total_loss += criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()

        # 记录测试信息
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / test_size
        print(f"测试集平均Loss: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
        writer.add_scalar("test_loss", avg_loss, epoch)
        writer.add_scalar("test_accuracy", accuracy, epoch)
        test_steps += 1

        # 保存模型
        torch.save(model.state_dict(), f"model_save/alex_epoch{epoch}.pth")
        print(f"模型已保存到: model_save/alex_epoch{epoch}.pth")

    # 6. 训练结束
    total_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {total_time:.2f}秒")
    writer.close()

if __name__ == "__main__":
    main()