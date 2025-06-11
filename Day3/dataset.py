import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Tuple

# Constants
DATASET_DIR = Path('Images')  # Using pathlib for better path handling
TRAIN_RATIO = 0.7
RANDOM_SEED = 42
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def setup_directories() -> Tuple[Path, Path]:
    """Create train and validation directories if they don't exist."""
    train_dir = DATASET_DIR / 'train'
    val_dir = DATASET_DIR / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    return train_dir, val_dir

def get_class_images(class_path: Path) -> List[Path]:
    """Get all valid images from a class directory."""
    return [img for img in class_path.iterdir() if img.suffix.lower() in VALID_EXTENSIONS]

def split_dataset() -> None:
    """Split dataset into train and validation sets."""
    random.seed(RANDOM_SEED)
    train_dir, val_dir = setup_directories()
    
    for class_dir in DATASET_DIR.iterdir():
        if class_dir.is_dir() and class_dir.name not in {'train', 'val'}:
            # Get and split images
            images = get_class_images(class_dir)
            train_imgs, val_imgs = train_test_split(
                images, train_size=TRAIN_RATIO, random_state=RANDOM_SEED
            )
            
            # Create class subdirectories
            (train_dir/class_dir.name).mkdir(exist_ok=True)
            (val_dir/class_dir.name).mkdir(exist_ok=True)
            
            # Move files
            for img in train_imgs:
                shutil.move(str(img), str(train_dir/class_dir.name/img.name))
            for img in val_imgs:
                shutil.move(str(img), str(val_dir/class_dir.name/img.name))
            
            # Remove original class directory
            shutil.rmtree(class_dir)

if __name__ == "__main__":
    try:
        split_dataset()
        print("Dataset successfully split into train and validation sets.")
    except Exception as e:
        print(f"Error occurred: {e}")