"""
Fundus preprocessing, step 1: crop the retina disk region out of a raw
fundus photo, then pad and resize to a square canvas without distorting
the aspect ratio (see Fig. 2 of the paper).
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image


def normalize_fundus(input_folder, output_folder, target_size=224):
    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(input_folder)
    print(f"{len(files)} images to process...")

    count = 0
    for filename in files:
        path = os.path.join(input_folder, filename)
        img_cv = cv2.imread(path)
        if img_cv is None:
            continue

        # Find the non-black (retina) region.
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)

        # Bounding box -> fundus diameter.
        x, y, w, h = cv2.boundingRect(coords)
        fundus_diameter = max(w, h)

        # Scale so the fundus diameter matches target_size.
        scale = target_size / fundus_diameter
        crop = img_cv[y:y + h, x:x + w]
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Center on a black target_size x target_size canvas.
        pad_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        pad_img[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        final_img = Image.fromarray(cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB))
        save_path = os.path.join(output_folder, filename.rsplit(".", 1)[0] + ".png")
        final_img.save(save_path)
        count += 1

    print(f"Done. {count} images processed and saved to '{output_folder}'.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_dir", help="Folder of raw fundus images.")
    p.add_argument("output_dir", help="Folder to write cropped/padded/resized images to.")
    p.add_argument("--target-size", type=int, default=224)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    normalize_fundus(args.input_dir, args.output_dir, args.target_size)
