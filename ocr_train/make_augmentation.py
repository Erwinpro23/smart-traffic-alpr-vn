"""Ghi ocr_train/augmentation.yaml: augmentation mặc định của fast-plate-ocr (rgb) + mô phỏng biển nhỏ, mờ trong video.

Ảnh được resize về 64x128 trước khi augment. Biển trong video 1080p rộng ~30-60 px, tức bị phóng 2-4 lần lên 128 px,
nên Downscale 0.25-0.6 tái tạo đúng mức mất chi tiết đó.
"""
from pathlib import Path

import albumentations as A
import cv2

BLACK = (0, 0, 0)


def build():
    return A.Compose([
        A.Affine(translate_percent=(-0.02, 0.02), scale=(0.85, 1.08), rotate=(-10, 10), shear=(-6, 6),
                 border_mode=cv2.BORDER_CONSTANT, fill=BLACK, p=0.6),
        A.Perspective(scale=(0.02, 0.06), p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.Downscale(scale_range=(0.25, 0.6),
                    interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.GaussianBlur(sigma_limit=(0.3, 1.2), p=1.0),
        ], p=0.35),
        A.ImageCompression(quality_range=(30, 90), p=0.4),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.OneOf([
            A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(1, 5), hole_width_range=(1, 5), p=1.0),
            A.PixelDropout(dropout_prob=0.02, p=1.0),
        ], p=0.15),
    ])


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "configs" / "augmentation.yaml"
    A.save(build(), str(out), data_format="yaml")
    print("ghi", out)
