import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from scipy import ndimage

try:
    from cellpose import models
except Exception as exc:  # pragma: no cover - surfaced directly in app UI
    models = None
    CELLPPOSE_IMPORT_ERROR = exc
else:
    CELLPPOSE_IMPORT_ERROR = None


st.set_page_config(page_title="Cellpose Fine-Tuning Studio", layout="wide")


@dataclass
class FileRecord:
    name: str
    sample_id: str
    channel_id: str
    bytes_data: bytes


def parse_name(filename: str, pattern: str) -> Tuple[str, str]:
    stem = Path(filename).stem
    if not pattern.strip():
        return stem, "ch0"

    match = re.match(pattern, stem)
    if not match:
        return stem, "ch0"

    groups = match.groupdict()
    sample_id = groups.get("sample", stem)
    channel_id = groups.get("channel", "ch0")
    return sample_id, channel_id


def collect_records(files: Iterable, pattern: str) -> List[FileRecord]:
    records: List[FileRecord] = []
    for uploaded in files:
        sample_id, channel_id = parse_name(uploaded.name, pattern)
        records.append(
            FileRecord(
                name=uploaded.name,
                sample_id=sample_id,
                channel_id=channel_id,
                bytes_data=uploaded.getvalue(),
            )
        )
    return records


def load_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return arr[:, :, :3]
    return arr.squeeze()


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., :3]

    img = image.astype(np.float32)
    img -= img.min()
    denom = img.max() if img.max() > 0 else 1.0
    img = (img / denom) * 255.0
    return img.astype(np.uint8)


def masks_to_overlay(image: np.ndarray, masks: np.ndarray, alpha: float, brush_size: int) -> np.ndarray:
    base = to_rgb_uint8(image)
    overlay = base.copy().astype(np.float32)
    max_label = int(masks.max())
    if max_label == 0:
        return base

    rng = np.random.default_rng(7)
    colors = rng.integers(40, 255, size=(max_label + 1, 3), dtype=np.uint8)

    tint = np.zeros_like(base, dtype=np.uint8)
    for label in range(1, max_label + 1):
        tint[masks == label] = colors[label]

    overlay = (1 - alpha) * overlay + alpha * tint.astype(np.float32)

    for label in range(1, max_label + 1):
        binary = masks == label
        if not binary.any():
            continue
        eroded = ndimage.binary_erosion(binary, iterations=max(1, brush_size))
        border = binary & ~eroded
        overlay[border] = (255, 255, 255)

    return np.clip(overlay, 0, 255).astype(np.uint8)


@st.cache_resource
def load_cellpose_model(model_type: str, gpu: bool):
    if models is None:
        raise RuntimeError(f"Cellpose import failed: {CELLPPOSE_IMPORT_ERROR}")
    return models.Cellpose(model_type=model_type, gpu=gpu)


def run_segmentation(model, image: np.ndarray, params: Dict) -> np.ndarray:
    masks, *_ = model.eval(
        image,
        diameter=params["diameter"],
        flow_threshold=params["flow_threshold"],
        cellprob_threshold=params["cellprob_threshold"],
        min_size=params["min_size"],
        normalize=params["normalize"],
        channels=[0, 0],
    )
    return masks


def read_mask(mask_bytes: bytes) -> np.ndarray:
    mask = np.asarray(Image.open(io.BytesIO(mask_bytes)))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32)


def main() -> None:
    st.title("Cellpose segmentation + fine-tuning app")
    st.caption("Upload image sets, review default overlays, tweak Cellpose parameters, and fine-tune a model.")

    if models is None:
        st.error(f"Cellpose is unavailable in this environment: {CELLPPOSE_IMPORT_ERROR}")
        st.stop()

    with st.sidebar:
        st.header("Model + overlay settings")
        model_type = st.selectbox("Base model", ["cyto", "cyto2", "cyto3", "nuclei"], index=0)
        use_gpu = st.checkbox("Use GPU", value=False)

        st.subheader("Segmentation parameters")
        diameter = st.number_input("Estimated diameter (px)", min_value=0.0, value=30.0, step=1.0)
        flow_threshold = st.slider("Flow threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        cellprob_threshold = st.slider("Cellprob threshold", min_value=-6.0, max_value=6.0, value=0.0, step=0.25)
        min_size = st.number_input("Min object size", min_value=0, value=15, step=1)
        normalize = st.checkbox("Normalize image", value=True)

        st.subheader("Overlay + brush settings")
        overlay_alpha = st.slider("Mask opacity", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
        brush_size = st.slider("Outline brush size (px)", min_value=1, max_value=12, value=2)

    st.markdown("### 1) Load image set")
    name_pattern = st.text_input(
        "Filename regex (optional)",
        value=r"(?P<sample>.+?)(?:_(?P<channel>ch\d+))?$",
        help="Use named groups `sample` and optional `channel` to group image sets.",
    )
    image_files = st.file_uploader(
        "Upload microscopy images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if not image_files:
        st.info("Upload one or more images to begin.")
        st.stop()

    records = collect_records(image_files, name_pattern)
    sample_index: Dict[str, List[FileRecord]] = {}
    for record in records:
        sample_index.setdefault(record.sample_id, []).append(record)

    sample_options = sorted(sample_index)
    selected_samples = st.multiselect(
        "Select sample(s) to review default segmentation overlays",
        sample_options,
        default=sample_options[: min(3, len(sample_options))],
    )

    if st.button("Run segmentation on selected sample(s)", type="primary"):
        model = load_cellpose_model(model_type=model_type, gpu=use_gpu)
        for sample in selected_samples:
            st.markdown(f"#### Sample: {sample}")
            cols = st.columns(len(sample_index[sample]))
            for idx, rec in enumerate(sorted(sample_index[sample], key=lambda r: r.channel_id)):
                img = load_image(rec.bytes_data)
                params = {
                    "diameter": diameter if diameter > 0 else None,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "min_size": int(min_size),
                    "normalize": normalize,
                }
                masks = run_segmentation(model, img, params)
                overlay = masks_to_overlay(img, masks, overlay_alpha, brush_size)
                cols[idx].image(overlay, caption=f"{rec.name} ({rec.channel_id})", use_container_width=True)
                cols[idx].write(f"Detected objects: **{int(masks.max())}**")

    st.markdown("---")
    st.markdown("### 2) Fine-tune Cellpose on custom labels")
    st.write("Upload matching images and label masks. Pairing is done by `sample` parsed from the same regex.")

    train_images = st.file_uploader(
        "Training images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="train_images",
    )
    train_masks = st.file_uploader(
        "Training masks (integer label maps)",
        type=["png", "tif", "tiff"],
        accept_multiple_files=True,
        key="train_masks",
    )

    col1, col2, col3 = st.columns(3)
    n_epochs = col1.number_input("Epochs", min_value=10, max_value=1000, value=100, step=10)
    learning_rate = col2.number_input("Learning rate", min_value=1e-6, max_value=1e-1, value=1e-3, format="%.6f")
    weight_decay = col3.number_input("Weight decay", min_value=0.0, max_value=1.0, value=1e-5, format="%.6f")
    save_dir = st.text_input("Output model directory", value="./trained_models")

    if st.button("Start fine-tuning"):
        if not train_images or not train_masks:
            st.error("Please provide both training images and masks.")
            st.stop()

        train_image_records = collect_records(train_images, name_pattern)
        train_mask_records = collect_records(train_masks, name_pattern)

        image_map = {r.sample_id: r for r in train_image_records}
        mask_map = {r.sample_id: r for r in train_mask_records}
        shared_samples = sorted(set(image_map) & set(mask_map))

        if not shared_samples:
            st.error("No matching image/mask sample IDs found. Check filename regex and inputs.")
            st.stop()

        train_data = [load_image(image_map[s].bytes_data) for s in shared_samples]
        train_labels = [read_mask(mask_map[s].bytes_data) for s in shared_samples]

        os.makedirs(save_dir, exist_ok=True)
        cp_model = load_cellpose_model(model_type=model_type, gpu=use_gpu)

        with st.spinner("Fine-tuning in progress..."):
            new_model_path, train_losses, test_losses = cp_model.train(
                train_data=train_data,
                train_labels=train_labels,
                channels=[0, 0],
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                n_epochs=int(n_epochs),
                save_path=save_dir,
                model_name=f"finetuned_{model_type}",
            )

        st.success(f"Fine-tuning complete. Model saved at: {new_model_path}")
        st.line_chart({"train_loss": train_losses, "test_loss": test_losses})


if __name__ == "__main__":
    main()
