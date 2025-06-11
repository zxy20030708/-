import os
from pathlib import Path
from typing import List, Tuple

def generate_dataset_index(root_dir: str, output_file: str) -> None:
    """Generate a dataset index file mapping image paths to labels.
    
    Args:
        root_dir: Directory containing category subdirectories.
        output_file: Path to the output text file.
        
    Raises:
        ValueError: If root_dir doesn't exist or is empty.
    """
    root_path = Path(root_dir)
    
    # Validate input directory
    if not root_path.exists():
        raise ValueError(f"Directory not found: {root_dir}")
    if not root_path.is_dir():
        raise ValueError(f"Path is not a directory: {root_dir}")
    
    # Get sorted categories to ensure consistent label ordering
    categories = sorted([d for d in os.listdir(root_path) 
                        if (root_path / d).is_dir()])
    if not categories:
        raise ValueError(f"No category directories found in: {root_dir}")
    
    # Write image paths with labels
    with open(output_file, 'w') as f:
        for label, category in enumerate(categories):
            category_path = root_path / category
            for img_name in sorted(os.listdir(category_path)):
                img_path = category_path / img_name
                f.write(f"{img_path} {label}\n")

def main() -> None:
    """Generate training and validation dataset index files."""
    try:
        generate_dataset_index('Images/train', 'train.txt')
        generate_dataset_index('Images/val', 'val.txt')
        print("Dataset index files created successfully.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()