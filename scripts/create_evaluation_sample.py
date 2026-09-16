"""
Script to initialize evaluation dataset structure and sample benchmark images.
"""

import csv
import os
from PIL import Image, ImageDraw, ImageFilter


def generate_sample_xray(filename: str, pattern_type: str) -> None:
    """Generate a realistic synthetic chest X-ray image for testing evaluation pipeline."""
    img = Image.new("L", (256, 256), color=15)
    draw = ImageDraw.Draw(img)

    # Draw thoracic cavity & lungs (darker regions)
    draw.ellipse([30, 40, 115, 220], fill=40)
    draw.ellipse([141, 40, 226, 220], fill=40)

    # Draw spine & ribs (lighter white structures)
    draw.rectangle([123, 20, 133, 250], fill=180)
    for y in range(60, 210, 20):
        draw.arc([40, y, 216, y + 30], start=190, end=350, fill=140, width=3)

    # Draw heart silhouette
    draw.ellipse([95, 130, 160, 210], fill=160)

    # Add specific pattern features based on finding type
    if pattern_type == "pneumonia":
        draw.ellipse([45, 110, 95, 160], fill=190)  # Focal opacity
    elif pattern_type == "pleural_effusion":
        draw.polygon([(30, 180), (115, 180), (115, 220), (30, 220)], fill=200)  # Fluid level
    elif pattern_type == "cardiomegaly":
        draw.ellipse([80, 120, 180, 225], fill=175)  # Enlarged heart shadow
    elif pattern_type == "pneumothorax":
        draw.line([(145, 50), (145, 200)], fill=240, width=2)  # Pleural line
    elif pattern_type == "pulmonary_edema":
        draw.ellipse([50, 80, 110, 140], fill=120)
        draw.ellipse([145, 80, 205, 140], fill=120)  # Batwing pattern
    elif pattern_type == "atelectasis":
        draw.polygon([(150, 150), (210, 170), (210, 190)], fill=190)  # Linear collapse
    elif pattern_type == "consolidation":
        draw.ellipse([150, 90, 200, 150], fill=210)  # Dense consolidation

    # Blur to emulate X-ray scatter
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img = img.convert("RGB")
    img.save(filename, format="PNG")


def main() -> None:
    eval_dir = os.path.join("data", "evaluation")
    img_dir = os.path.join(eval_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(eval_dir, "labels.csv")

    headers = [
        "image_id",
        "split",
        "No Finding",
        "Pneumonia",
        "Effusion",
        "Pneumothorax",
        "Cardiomegaly",
        "Atelectasis",
        "Edema",
        "Consolidation",
    ]

    samples = [
        ("eval_001.png", "val", 1, 0, 0, 0, 0, 0, 0, 0, "normal"),
        ("eval_002.png", "val", 0, 1, 0, 0, 0, 0, 0, 0, "pneumonia"),
        ("eval_003.png", "val", 0, 0, 1, 0, 0, 0, 0, 0, "pleural_effusion"),
        ("eval_004.png", "val", 0, 0, 0, 1, 0, 0, 0, 0, "pneumothorax"),
        ("eval_005.png", "val", 0, 0, 0, 0, 1, 0, 0, 0, "cardiomegaly"),
        ("eval_006.png", "val", 0, 0, 0, 0, 0, 1, 0, 0, "atelectasis"),
        ("eval_007.png", "val", 0, 0, 0, 0, 0, 0, 1, 0, "pulmonary_edema"),
        ("eval_008.png", "val", 0, 0, 0, 0, 0, 0, 0, 1, "consolidation"),
        ("eval_009.png", "test", 1, 0, 0, 0, 0, 0, 0, 0, "normal"),
        ("eval_010.png", "test", 0, 1, 0, 0, 0, 0, 0, 0, "pneumonia"),
        ("eval_011.png", "test", 0, 0, 1, 0, 0, 0, 0, 0, "pleural_effusion"),
        ("eval_012.png", "test", 0, 0, 0, 1, 0, 0, 0, 0, "pneumothorax"),
        ("eval_013.png", "test", 0, 0, 0, 0, 1, 0, 0, 0, "cardiomegaly"),
        ("eval_014.png", "test", 0, 0, 0, 0, 0, 1, 0, 0, "atelectasis"),
        ("eval_015.png", "test", 0, 0, 0, 0, 0, 0, 1, 0, "pulmonary_edema"),
        ("eval_016.png", "test", 0, 0, 0, 0, 0, 0, 0, 1, "consolidation"),
    ]

    rows = []
    for item in samples:
        img_id, split, nf, pne, eff, pno, car, ate, ede, con, ptype = item
        img_path = os.path.join(img_dir, img_id)
        generate_sample_xray(img_path, ptype)
        rows.append([img_id, split, nf, pne, eff, pno, car, ate, ede, con])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Successfully generated {len(samples)} evaluation images in '{img_dir}'")
    print(f"Ground truth labels written to '{csv_path}'")


if __name__ == "__main__":
    main()
