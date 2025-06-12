import time
import os
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Chen
from dataset import ImageTxtDataset

class Trainer:
    """Complete training pipeline for CIFAR10-like datasets."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_directories()
        self._setup_datasets()
        self._setup_model()
        self.writer = SummaryWriter(self.config['log_dir'])
        
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        Path(self.config['model_save_dir']).mkdir(parents=True, exist_ok=True)
        
    def _setup_datasets(self) -> None:
        """Initialize datasets and data loaders."""
        # Training transforms
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms
        val_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize datasets
        self.train_data = ImageTxtDataset(
            self.config['train_txt'],
            self.config['train_img_dir'],
            train_transform
        )
        self.test_data = ImageTxtDataset(
            self.config['val_txt'],
            self.config['val_img_dir'],
            val_transform
        )
        
        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_data, 
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_data, 
            batch_size=self.config['batch_size']
        )
        
        print(f"训练数据集长度: {len(self.train_data)}")
        print(f"测试数据集长度: {len(self.test_data)}")
        
    def _setup_model(self) -> None:
        """Initialize model, loss function and optimizer."""
        self.model = Chen().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        
    def train_epoch(self, epoch: int) -> None:
        """Train for one epoch."""
        self.model.train()
        for batch_idx, (imgs, targets) in enumerate(self.train_loader):
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(imgs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Log training metrics
            if batch_idx % 500 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
                self.writer.add_scalar("train_loss", loss.item(), epoch * len(self.train_loader) + batch_idx)
    
    def evaluate(self, epoch: int) -> Tuple[float, float]:
        """Evaluate model on test set."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        
        with torch.no_grad():
            for imgs, targets in self.test_loader:
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == targets).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = total_correct / len(self.test_data)
        
        # Log evaluation metrics
        self.writer.add_scalar("test_loss", avg_loss, epoch)
        self.writer.add_scalar("test_accuracy", accuracy, epoch)
        
        return avg_loss, accuracy
    
    def save_model(self, epoch: int) -> None:
        """Save model checkpoint."""
        model_path = os.path.join(
            self.config['model_save_dir'],
            f"chen_{epoch}.pth"
        )
        torch.save(self.model.state_dict(), model_path)
        print(f"模型已保存: {model_path}")
        
    def run(self) -> None:
        """Run complete training pipeline."""
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            print(f"\n----- 第 {epoch + 1}/{self.config['epochs']} 轮训练开始 -----")
            
            # Training phase
            epoch_start = time.time()
            self.train_epoch(epoch)
            
            # Evaluation phase
            test_loss, test_acc = self.evaluate(epoch)
            print(f"测试集平均Loss: {test_loss:.4f}, 准确率: {test_acc:.4f}")
            
            # Save model
            self.save_model(epoch)
            
            print(f"本轮训练时间: {time.time() - epoch_start:.2f}秒")
        
        total_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {total_time:.2f}秒")
        self.writer.close()

if __name__ == "__main__":
    # Configuration
    config = {
        'dataset_root': r"E:\pythonproject\dateset",
        'train_txt': "train.txt",
        'val_txt': "val.txt",
        'train_img_dir': os.path.join("Images", "train"),
        'val_img_dir': os.path.join("Images", "val"),
        'batch_size': 64,
        'learning_rate': 0.01,
        'epochs': 10,
        'log_dir': "../logs_train",
        'model_save_dir': "model_save"
    }
    
    # Build absolute paths
    config['train_txt'] = os.path.join(config['dataset_root'], config['train_txt'])
    config['val_txt'] = os.path.join(config['dataset_root'], config['val_txt'])
    config['train_img_dir'] = os.path.join(config['dataset_root'], config['train_img_dir'])
    config['val_img_dir'] = os.path.join(config['dataset_root'], config['val_img_dir'])
    
    # Run training
    trainer = Trainer(config)
    trainer.run()