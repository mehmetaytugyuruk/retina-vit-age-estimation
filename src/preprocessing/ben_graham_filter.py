"""
Fundus preprocessing, step 2 (optional): Ben Graham's filter, as used in
the Kaggle diabetic retinopathy detection competition (Graham, 2015).
Sharpens vessel structures via a weighted local-contrast transform.

Note: despite historical naming in earlier drafts of this code, this
module does NOT implement CLAHE -- only the Ben Graham weighting.
"""

import argparse
import os

import cv2

SIGMA_X = 10  # Gaussian blur sigma; controls how strongly vessels are emphasized.


def apply_ben_graham_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold at 7 to exclude near-black background while keeping dim retina detail.
    _, mask = cv2.threshold(gray, 7, 255, cv2.THRESH_BINARY)

    # Ben Graham method: 4 * original - 4 * blurred + 128.
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=SIGMA_X)
    processed = cv2.addWeighted(image, 4, blurred, -4, 128)

    # Re-apply the retina mask so the background stays black, not gray.
    return cv2.bitwise_and(processed, processed, mask=mask)


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    print(f"{len(files)} images to process...")

    count = 0
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        try:
            img = cv2.imread(input_path)
            if img is None:
                print(f"WARNING: could not read {filename}, skipping.")
                continue

            processed_img = apply_ben_graham_filter(img)
            cv2.imwrite(output_path, processed_img)
            count += 1
            if count % 50 == 0:
                print(f"{count} images done...")
        except Exception as e:
            print(f"ERROR processing {filename}: {e}")

    print(f"Done. {count}/{len(files)} files saved to '{output_folder}'.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_dir", help="Folder of cropped/resized fundus images.")
    p.add_argument("output_dir", help="Folder to write Graham-filtered images to.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_folder(args.input_dir, args.output_dir)
