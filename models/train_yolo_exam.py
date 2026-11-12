"""
YOLOv8 Examination Sheet Custom Object Detector Training Script
==============================================================
This script provides the pipeline for training a custom YOLOv8 model to detect
examination answer sheet fields (Student Name, Enrollment No, Marks Table, etc.)
if the institution switches from fixed booklet templates to variable unstructured layouts.

WHEN TO USE THIS SCRIPT:
------------------------
- Currently, the exam answer booklets use a standardized template with printed headers.
  Perspective correction + spatial label anchoring delivers 95%+ precision without
  requiring an annotated bounding box dataset.
- Train this custom YOLO model ONLY IF:
  1. The institution introduces non-standard, multi-format booklet designs with variable positions.
  2. The booklet layout shifts unpredictably where spatial anchors cannot be detected.

DATASET FORMAT REQUIREMENTS:
----------------------------
The dataset follows standard YOLO format:
  dataset/
    ├── exam_dataset.yaml
    ├── images/
    │   ├── train/  (*.jpg / *.png)
    │   └── val/    (*.jpg / *.png)
    └── labels/
        ├── train/  (*.txt, one per image)
        └── val/    (*.txt, one per image)

Each .txt annotation file contains lines in YOLO normalized coordinates:
  <class_id> <x_center> <y_center> <width> <height>
  (all values normalized between 0.0 and 1.0)

CLASS MAPPING (10 Classes):
---------------------------
  0: student_name
  1: enrollment_no
  2: test_no
  3: section
  4: exam_date
  5: class_year
  6: course_title
  7: course_code
  8: marks_table
  9: grand_total
"""

import os
import sys
import yaml
from pathlib import Path

def create_dataset_yaml(output_path: str, data_root: str):
    """Generates standard YOLOv8 dataset configuration file."""
    dataset_cfg = {
        'path': os.path.abspath(data_root),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'student_name',
            1: 'enrollment_no',
            2: 'test_no',
            3: 'section',
            4: 'exam_date',
            5: 'class_year',
            6: 'course_title',
            7: 'course_code',
            8: 'marks_table',
            9: 'grand_total'
        }
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)
    print(f"[YOLO TRAIN] Generated dataset YAML configuration at: {output_path}")

def train_exam_yolo(
    data_yaml_path: str,
    epochs: int = 50,
    img_size: int = 1280,
    batch_size: int = 4,
    pretrained_weights: str = 'yolov8n.pt',
    output_dir: str = 'runs/exam_detect'
):
    """
    Fine-tunes YOLOv8 on annotated examination sheets.
    Requires: pip install ultralytics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"==================================================")
    print(f"Starting YOLOv8 Exam Field Training")
    print(f"Base Weights: {pretrained_weights}")
    print(f"Image Size:   {img_size}x{img_size}")
    print(f"Epochs:       {epochs}")
    print(f"Batch Size:   {batch_size}")
    print(f"Config YAML:  {data_yaml_path}")
    print(f"==================================================")

    model = YOLO(pretrained_weights)

    # Train model with augmentations tuned for scanned/mobile documents
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=output_dir,
        name='exam_model',
        degrees=10.0,       # Random slight rotation
        shear=2.0,          # Minor perspective shear
        perspective=0.0005, # Camera perspective simulation
        flipud=0.0,         # Never flip upside down
        fliplr=0.0,         # Never flip left-right (text readability)
        mosaic=0.5,         # Mild mosaic augmentation
        mixup=0.0,          # Avoid mixing documents
        workers=2,
        exist_ok=True
    )

    best_model_path = os.path.join(output_dir, 'exam_model', 'weights', 'best.pt')
    target_path = os.path.join(os.path.dirname(__file__), 'exam_yolov8.pt')
    if os.path.exists(best_model_path):
        import shutil
        shutil.copyfile(best_model_path, target_path)
        print(f"[SUCCESS] Trained exam weights saved to: {target_path}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train custom YOLOv8 model for examination sheet field detection")
    parser.add_argument("--data", type=str, default="data/exam_dataset.yaml", help="Path to exam_dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1280, help="Training image size (1280 recommended for text)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[INFO] Dataset config {args.data} does not exist yet.")
        print("To train a custom YOLO model:")
        print("1. Label ~100-200 booklet images with CVAT, LabelImg, or Roboflow using classes 0-9.")
        print("2. Organize images and labels into images/train, labels/train, images/val, labels/val.")
        print(f"3. Run: python models/train_yolo_exam.py --data {args.data}")
    else:
        train_exam_yolo(args.data, epochs=args.epochs, img_size=args.imgsz, batch_size=args.batch)
