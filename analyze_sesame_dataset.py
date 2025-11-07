"""
Analyze Sesame_Dataset for fine-tuning YOLOv5 model
"""

import os
from pathlib import Path
import yaml

def analyze_sesame_dataset():
    """Analyze the Sesame_Dataset structure and annotations"""
    
    print("="*70)
    print("🔍 ANALYZING SESAME_DATASET")
    print("="*70)
    
    # Paths
    dataset_root = Path("Sesame_Dataset")
    data_dir = dataset_root / "agri_data" / "data"
    classes_file = dataset_root / "classes.txt"
    
    # Read classes
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    print(f"\n📋 Classes: {classes}")
    print(f"   Total classes: {len(classes)}")
    
    # Count images
    images = list(data_dir.glob("*.jpeg"))
    labels = list(data_dir.glob("*.txt"))
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images: {len(images)}")
    print(f"   Total labels: {len(labels)}")
    
    # Analyze annotations
    crop_count = 0
    weed_count = 0
    total_objects = 0
    images_with_crop = 0
    images_with_weed = 0
    images_with_both = 0
    
    for label_file in labels:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        has_crop = False
        has_weed = False
        
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    total_objects += 1
                    
                    if class_id == 0:
                        crop_count += 1
                        has_crop = True
                    elif class_id == 1:
                        weed_count += 1
                        has_weed = True
        
        if has_crop:
            images_with_crop += 1
        if has_weed:
            images_with_weed += 1
        if has_crop and has_weed:
            images_with_both += 1
    
    print(f"\n📊 Object Distribution:")
    print(f"   Total objects: {total_objects}")
    print(f"   Crop objects: {crop_count} ({crop_count/total_objects*100:.1f}%)")
    print(f"   Weed objects: {weed_count} ({weed_count/total_objects*100:.1f}%)")
    print(f"   Imbalance ratio: {max(crop_count, weed_count) / min(crop_count, weed_count):.2f}:1")
    
    print(f"\n🖼️  Image Distribution:")
    print(f"   Images with crops: {images_with_crop}")
    print(f"   Images with weeds: {images_with_weed}")
    print(f"   Images with both: {images_with_both}")
    
    # Check if train/val/test split exists
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    
    if train_dir.exists() or val_dir.exists() or test_dir.exists():
        print(f"\n✅ Dataset has train/val/test split")
    else:
        print(f"\n⚠️  Dataset does NOT have train/val/test split")
        print(f"   All images are in one folder: {data_dir}")
        print(f"   Need to create train/val/test split")
    
    print("\n" + "="*70)
    print("📋 RECOMMENDATIONS:")
    print("="*70)
    
    if crop_count == 0 or weed_count == 0:
        print("❌ Missing one of the classes! Check annotations.")
    else:
        imbalance = max(crop_count, weed_count) / min(crop_count, weed_count)
        if imbalance > 3:
            print("⚠️  Severe class imbalance - consider:")
            print("   - Data augmentation")
            print("   - Class weighting")
            print("   - Collecting more minority class data")
    
    print("\n✅ Next steps:")
    print("1. Create train/val/test split (if needed)")
    print("2. Organize into YOLOv5 format:")
    print("   Sesame_Dataset/")
    print("   ├── train/")
    print("   │   ├── images/")
    print("   │   └── labels/")
    print("   ├── val/")
    print("   │   ├── images/")
    print("   │   └── labels/")
    print("   └── test/")
    print("       ├── images/")
    print("       └── labels/")
    print("3. Create data.yaml for YOLOv5")
    print("4. Fine-tune your 97% model!")
    
    print("\n" + "="*70)
    
    return {
        'total_images': len(images),
        'total_labels': len(labels),
        'total_objects': total_objects,
        'crop_count': crop_count,
        'weed_count': weed_count,
        'classes': classes,
        'has_split': (train_dir.exists() or val_dir.exists() or test_dir.exists())
    }

if __name__ == "__main__":
    analyze_sesame_dataset()
