#!/usr/bin/env python3
"""Export a single LeRobot episode parquet to an Excel workbook with one frame per row."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any


def _require_dependency(module_name: str, package_hint: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency '{module_name}'. Install it in the export environment first, for example: {package_hint}"
        ) from exc


pd = _require_dependency("pandas", "pip install pandas pyarrow openpyxl pillow opencv-python")
np = _require_dependency("numpy", "pip install numpy")
cv2 = _require_dependency("cv2", "pip install opencv-python")
openpyxl = _require_dependency("openpyxl", "pip install openpyxl")
PIL_Image = _require_dependency("PIL.Image", "pip install pillow")
OpenpyxlImage = _require_dependency("openpyxl.drawing.image", "pip install openpyxl").Image

ACTION_DIM_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


def _load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return {}
    return json.loads(info_path.read_text(encoding="utf-8"))


def _extract_episode_index(episode_path: Path, df) -> int:
    match = re.search(r"episode_(\d+)\.parquet$", episode_path.name)
    if match:
        return int(match.group(1))
    if "episode_index" in df.columns:
        return int(df.iloc[0]["episode_index"])
    raise ValueError(f"Could not infer episode index from {episode_path}")


def _infer_dataset_root(episode_path: Path) -> Path:
    current = episode_path.resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "meta" / "info.json").is_file():
            return parent
    raise FileNotFoundError(
        f"Could not find dataset root for {episode_path}. Expected a parent directory containing meta/info.json."
    )


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, bytes, int, float, bool)) or getattr(value, "shape", None) == ()


def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _is_image_column(name: str, value: Any) -> bool:
    if "image" not in name.lower():
        return False

    if isinstance(value, bytes):
        return True
    if hasattr(value, "mode") and hasattr(value, "size"):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim in (2, 3)
    if isinstance(value, Mapping):
        return "bytes" in value or "path" in value
    return False


def _flatten_cell(prefix: str, value: Any) -> dict[str, Any]:
    value = _normalize_scalar(value)

    if value is None or _is_scalar(value):
        return {prefix: value}

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return {prefix: value.item()}
        if value.ndim == 1:
            if value.shape[0] == 1:
                return {prefix: _normalize_scalar(value[0])}
            return {prefix: json.dumps(np.asarray(value).tolist(), ensure_ascii=False)}
        return {prefix: json.dumps(np.asarray(value).tolist(), ensure_ascii=False)}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) == 1:
            return {prefix: _normalize_scalar(value[0])}
        return {prefix: json.dumps(list(value), ensure_ascii=False, default=str)}

    if isinstance(value, Mapping):
        return {prefix: json.dumps(dict(value), ensure_ascii=False, default=str)}

    return {prefix: str(value)}


def _flatten_dataframe(df) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    image_columns: list[str] = []
    ordered_columns: list[str] = []

    sample_row = df.iloc[0] if len(df) else None
    for column in df.columns:
        if sample_row is not None and _is_image_column(column, sample_row[column]):
            image_columns.append(column)

    for _, row in df.iterrows():
        flat_row: dict[str, Any] = {}
        for column in df.columns:
            if column in image_columns:
                flat_row[column] = None
                continue
            for flat_key, flat_value in _flatten_cell(column, row[column]).items():
                flat_row[flat_key] = flat_value
                if flat_key not in ordered_columns:
                    ordered_columns.append(flat_key)
        rows.append(flat_row)

    return rows, ordered_columns, image_columns


def _load_rgb_frame(frame: Any):
    if frame is None:
        return None

    if isinstance(frame, bytes):
        with BytesIO(frame) as buffer:
            return PIL_Image.open(buffer).convert("RGB")

    if isinstance(frame, Mapping):
        if frame.get("path"):
            return PIL_Image.open(frame["path"]).convert("RGB")
        if frame.get("bytes"):
            with BytesIO(frame["bytes"]) as buffer:
                return PIL_Image.open(buffer).convert("RGB")

    if hasattr(frame, "mode") and hasattr(frame, "size"):
        return frame.convert("RGB")

    if isinstance(frame, np.ndarray):
        array = np.asarray(frame)
        if array.ndim == 2:
            return PIL_Image.fromarray(array).convert("RGB")
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        return PIL_Image.fromarray(array.astype(np.uint8)).convert("RGB")

    raise TypeError(f"Unsupported image payload type: {type(frame)}")


def _export_images_from_dataframe(df, image_columns: list[str], image_dir: Path) -> dict[str, list[Path | None]]:
    exported: dict[str, list[Path | None]] = {}
    for column in image_columns:
        column_dir = image_dir / column
        column_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path | None] = []
        for row_idx, frame in enumerate(df[column].tolist()):
            if frame is None:
                paths.append(None)
                continue
            image = _load_rgb_frame(frame)
            output_path = column_dir / f"frame_{row_idx:06d}.png"
            image.save(output_path)
            paths.append(output_path)
        exported[column] = paths
    return exported


def _export_images_from_videos(
    dataset_root: Path,
    info: dict[str, Any],
    episode_index: int,
    frame_count: int,
    image_dir: Path,
) -> dict[str, list[Path | None]]:
    features = info.get("features", {})
    video_columns = [
        name for name, spec in features.items() if spec.get("dtype") in {"video", "image"} and name.startswith("observation.images.")
    ]
    if not video_columns:
        return {}

    video_template = info.get("video_path")
    if not video_template:
        return {}

    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunk_size
    exported: dict[str, list[Path | None]] = {}

    for column in video_columns:
        video_path = dataset_root / video_template.format(episode_chunk=episode_chunk, video_key=column, episode_index=episode_index)
        if not video_path.is_file():
            continue

        column_dir = image_dir / column
        column_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        frame_paths: list[Path | None] = []
        for frame_idx in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                frame_paths.extend([None] * (frame_count - frame_idx))
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_path = column_dir / f"frame_{frame_idx:06d}.png"
            PIL_Image.fromarray(rgb).save(output_path)
            frame_paths.append(output_path)

        capture.release()

        if len(frame_paths) < frame_count:
            frame_paths.extend([None] * (frame_count - len(frame_paths)))

        exported[column] = frame_paths

    return exported


def _set_basic_layout(sheet, header_count: int, row_count: int) -> None:
    sheet.freeze_panes = "A2"
    for idx in range(1, header_count + 1):
        sheet.column_dimensions[openpyxl.utils.get_column_letter(idx)].width = 16
    for row_idx in range(2, row_count + 2):
        sheet.row_dimensions[row_idx].height = 90


def _write_action_readme_sheet(workbook) -> None:
    sheet = workbook.create_sheet("README")
    sheet.append(["action dimension", "meaning"])
    for idx, name in enumerate(ACTION_DIM_NAMES):
        sheet.append([f"action[{idx}]", name])
    sheet.column_dimensions["A"].width = 18
    sheet.column_dimensions["B"].width = 24


def export_episode_to_excel(
    episode_path: str | Path,
    output_path: str | Path | None = None,
    image_width_px: int = 160,
) -> Path:
    episode_path = Path(episode_path)
    if not episode_path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {episode_path}")

    df = pd.read_parquet(episode_path)
    if df.empty:
        raise ValueError(f"Episode parquet is empty: {episode_path}")

    dataset_root = _infer_dataset_root(episode_path)
    info = _load_info(dataset_root)
    episode_index = _extract_episode_index(episode_path, df)
    output_path = Path(output_path) if output_path else episode_path.with_suffix(".xlsx")

    flattened_rows, scalar_columns, embedded_image_columns = _flatten_dataframe(df)

    with tempfile.TemporaryDirectory(prefix=f"{episode_path.stem}_excel_") as tmp_dir:
        temp_image_dir = Path(tmp_dir)
        image_paths = _export_images_from_dataframe(df, embedded_image_columns, temp_image_dir)

        if not image_paths:
            image_paths = _export_images_from_videos(
                dataset_root=dataset_root,
                info=info,
                episode_index=episode_index,
                frame_count=len(df),
                image_dir=temp_image_dir,
            )

        image_columns = list(image_paths.keys())
        headers = scalar_columns + image_columns

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = episode_path.stem
        sheet.append(headers)

        for row_idx, flat_row in enumerate(flattened_rows, start=2):
            row_values = [flat_row.get(column) for column in scalar_columns]
            row_values.extend([None] * len(image_columns))
            sheet.append(row_values)

            for image_col_idx, image_column in enumerate(image_columns, start=len(scalar_columns) + 1):
                frame_paths = image_paths.get(image_column, [])
                image_path = frame_paths[row_idx - 2] if row_idx - 2 < len(frame_paths) else None
                if image_path is None:
                    continue

                image_bytes = image_path.read_bytes()
                image_stream = BytesIO(image_bytes)
                image_stream.name = image_path.name
                image = OpenpyxlImage(image_stream)
                if image.width > image_width_px:
                    scale = image_width_px / image.width
                    image.width = int(image.width * scale)
                    image.height = int(image.height * scale)
                sheet.add_image(image, f"{openpyxl.utils.get_column_letter(image_col_idx)}{row_idx}")

        _set_basic_layout(sheet, len(headers), len(flattened_rows))
        _write_action_readme_sheet(workbook)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        workbook.save(output_path)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a single LeRobot episode parquet to an Excel workbook.")
    parser.add_argument("episode_path", type=Path, help="Path to a single episode parquet file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .xlsx path. Defaults to the episode path with .xlsx suffix.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=160,
        help="Maximum width of embedded images in pixels.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = export_episode_to_excel(
        episode_path=args.episode_path,
        output_path=args.output,
        image_width_px=args.image_width,
    )
    print(f"Saved Excel workbook to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
