import torch
import torchvision
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from typing import Tuple

# Constants
DATASET_ROOT = "dataset_chen"
BATCH_SIZE = 64
LOG_DIR = "sigmoid_logs"  # Fixed typo from original ("sigmod_logs")

class SigmoidNetwork(nn.Module):
    """Neural network that applies sigmoid activation.
    
    Maintains the exact same behavior as the original Chen class.
    """
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(x)

def load_dataset() -> Tuple[DataLoader, int]:
    """Load and prepare the CIFAR10 dataset.
    
    Returns:
        Tuple of (dataloader, dataset_length)
    """
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(
        root=DATASET_ROOT,
        train=False,
        transform=transform,
        download=True
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE), len(dataset)

def process_test_input() -> Tuple[Tensor, Tensor]:
    """Process the test input tensor (same as original)."""
    input_tensor = torch.tensor([[1, -0.5], [-1, 3]], dtype=torch.float32)
    return input_tensor.reshape(-1, 1, 2, 2), input_tensor

def main() -> None:
    """Main execution function."""
    # Initialize components
    model = SigmoidNetwork()
    dataloader, dataset_size = load_dataset()
    test_input, original_input = process_test_input()
    
    print(f"Test input shape: {test_input.shape}")
    
    # TensorBoard logging
    with SummaryWriter(LOG_DIR) as writer:
        for step, (imgs, targets) in enumerate(dataloader):
            # Process and log images
            writer.add_images("input", imgs, global_step=step)
            output = model(imgs)
            writer.add_images("output", output, global_step=step)
            
            # Early stop for demonstration (optional)
            if step > 10:  # Limit steps for quicker testing
                break
    
    # Process and print test input
    test_output = model(test_input)
    print("\nOriginal test input:")
    print(original_input)
    print("\nAfter sigmoid activation:")
    print(test_output)

if __name__ == "__main__":
    main()