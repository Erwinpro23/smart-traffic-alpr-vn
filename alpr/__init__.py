"""ALPR Vietnamese: Core Package for Vietnamese License Plate and Vehicle Recognition."""

from .pipeline import ALPRPipeline, PlateConfig, FrameResult, best_of, position_vote, resolve_type, associate_plates
from .vehicle_module import VehicleTracker, Track, TypeVoter, COARSE_CLASSES
from .plate_preprocess import enhance, rectify, sharpness, split_two_line, find_quad, order_corners
from .plate_text import clean, correct, is_valid, n_changes
from .fusion import (
    FusionConfig, PlateBuffer, aspect_type, box_touches_border,
    group_ids_by_plate, is_single_line, owner_of, single_line_type, expand_box
)
from .draw import annotate, draw_hud_text
from .alpr_common import (
    load_ocr, load_plate_model, normalize_plate, clean_text,
    get_expanded_crop, plate_model_candidates, plate_vehicle_class,
    setup_cuda_dlls, SCRIPT_DIR, DEMO_ROI, DEMO_VIDEO
)

__all__ = [
    "ALPRPipeline", "PlateConfig", "FrameResult", "best_of", "position_vote", "resolve_type", "associate_plates",
    "VehicleTracker", "Track", "TypeVoter", "COARSE_CLASSES",
    "enhance", "rectify", "sharpness", "split_two_line", "find_quad", "order_corners",
    "clean", "correct", "is_valid", "n_changes",
    "FusionConfig", "PlateBuffer", "aspect_type", "box_touches_border",
    "group_ids_by_plate", "is_single_line", "owner_of", "single_line_type", "expand_box",
    "annotate", "draw_hud_text",
    "load_ocr", "load_plate_model", "normalize_plate", "clean_text",
    "get_expanded_crop", "plate_model_candidates", "plate_vehicle_class",
    "setup_cuda_dlls", "SCRIPT_DIR", "DEMO_ROI", "DEMO_VIDEO"
]
