from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

import cv2
import numpy as np
import pyzed.sl as sl

OUTPUT_DIR = Path(__file__).with_name("video_captures")
FILE_STEM = "zed2i_video"
RECORD_SECONDS = 3
START_COUNTDOWN_SECONDS = 3
WARMUP_GRABS = 0
CAMERA_FPS = 30

CONFIDENCE_THRESHOLD = 95
TEXTURE_CONFIDENCE_THRESHOLD = 100
REMOVE_SATURATED_AREAS = False
ENABLE_FILL_MODE = False
DEPTH_STABILIZATION = 30

DEPTH_MINIMUM_DISTANCE_METERS = 0.5
DEPTH_MAXIMUM_DISTANCE_METERS = 4.0

AEC_AGC_ROI_X = 300
AEC_AGC_ROI_Y = 200
AEC_AGC_ROI_WIDTH = 2208 - 300 - 300
AEC_AGC_ROI_HEIGHT = 1242 - 200 - 200

DEPTH_MODE = sl.DEPTH_MODE.NEURAL_PLUS


def next_capture_dir(output_dir: Path, file_stem: str) -> Path:
    index = 1
    while (output_dir / f"{file_stem}_{index:03d}").exists():
        index += 1
    return output_dir / f"{file_stem}_{index:03d}"


def to_bgr(image: sl.Mat):
    data = image.get_data()
    if len(data.shape) == 3 and data.shape[2] == 4:
        return cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    return data


def open_writer(path: Path, width: int, height: int) -> cv2.VideoWriter:
    # ponytail: keep one codec choice here; switch container/codec only if this OpenCV build rejects mp4v.
    writer = cv2.VideoWriter(
        path.as_posix(),
        cv2.VideoWriter.fourcc(*"mp4v"),
        CAMERA_FPS,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Open video writer failed: {path}")
    return writer


def build_split_frame(left_frame, depth_frame):
    split_frame = cv2.hconcat([left_frame, depth_frame])
    cv2.putText(
        split_frame,
        "LEFT",
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        split_frame,
        "DEPTH 2D",
        (left_frame.shape[1] + 40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return split_frame


def depth_meters_to_png_mm(depth_meters: np.ndarray) -> np.ndarray:
    depth_plane = depth_meters[..., 0] if depth_meters.ndim == 3 else depth_meters
    depth_mm = np.zeros(depth_plane.shape, dtype=np.uint16)
    valid = np.isfinite(depth_plane) & (depth_plane > 0)
    depth_mm[valid] = np.clip(
        np.rint(depth_plane[valid] * 1000.0),
        1,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    return depth_mm


def main() -> int:
    if RECORD_SECONDS <= 0:
        print("RECORD_SECONDS must be > 0")
        return 1

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = next_capture_dir(output_dir, FILE_STEM)
    capture_dir.mkdir()

    left_frames_dir = capture_dir / "left_frames"
    right_frames_dir = capture_dir / "right_frames"
    depth_frames_dir = capture_dir / "depth_2d_frames"
    depth_metric_frames_dir = capture_dir / "depth_metric_mm_frames"
    split_frames_dir = capture_dir / "split_frames"
    left_frames_dir.mkdir()
    right_frames_dir.mkdir()
    depth_frames_dir.mkdir()
    depth_metric_frames_dir.mkdir()
    split_frames_dir.mkdir()

    left_video_path = capture_dir / f"{capture_dir.name}_left.mp4"
    right_video_path = capture_dir / f"{capture_dir.name}_right.mp4"
    depth_video_path = capture_dir / f"{capture_dir.name}_depth_2d.mp4"
    split_video_path = capture_dir / f"{capture_dir.name}_split_left_depth.mp4"

    # ponytail: fixed-fps frame budget keeps the saved video length deterministic; use a wall-clock stop only if dropped-frame timing matters more.
    frame_count = max(1, int(RECORD_SECONDS * CAMERA_FPS))
    init_params = sl.InitParameters()
    init_params.enable_image_enhancement = False
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = CAMERA_FPS
    init_params.depth_mode = DEPTH_MODE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_stabilization = DEPTH_STABILIZATION
    init_params.depth_minimum_distance = DEPTH_MINIMUM_DISTANCE_METERS
    init_params.depth_maximum_distance = DEPTH_MAXIMUM_DISTANCE_METERS
    init_params.sdk_verbose = 0

    runtime_params = sl.RuntimeParameters()
    runtime_params.enable_depth = True
    runtime_params.confidence_threshold = CONFIDENCE_THRESHOLD
    runtime_params.texture_confidence_threshold = TEXTURE_CONFIDENCE_THRESHOLD
    runtime_params.remove_saturated_areas = REMOVE_SATURATED_AREAS
    runtime_params.enable_fill_mode = ENABLE_FILL_MODE

    zed = sl.Camera()
    err = zed.open(init_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print(f"Camera open failed: {err}")
        return 1

    zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 4)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE, 0)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 4)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 4)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 5)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
    roi = sl.Rect(AEC_AGC_ROI_X, AEC_AGC_ROI_Y, AEC_AGC_ROI_WIDTH, AEC_AGC_ROI_HEIGHT)
    zed.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, roi, sl.SIDE.BOTH)

    resolution = zed.get_camera_information().camera_configuration.resolution
    width = resolution.width
    height = resolution.height

    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_image = sl.Mat()
    depth_measure = sl.Mat()
    left_writer = None
    right_writer = None
    depth_writer = None
    split_writer = None

    try:
        left_writer = open_writer(left_video_path, width, height)
        right_writer = open_writer(right_video_path, width, height)
        depth_writer = open_writer(depth_video_path, width, height)
        split_writer = open_writer(split_video_path, width * 2, height)

        for _ in range(WARMUP_GRABS):
            err = zed.grab(runtime_params)
            if err > sl.ERROR_CODE.SUCCESS:
                print(f"Warmup grab failed: {err}")
                return 1

        for seconds_left in range(START_COUNTDOWN_SECONDS, 0, -1):
            print(f"Recording starts in {seconds_left}...")
            sleep(1)
        print("Recording.")

        for frame_index in range(frame_count):
            err = zed.grab(runtime_params)
            if err > sl.ERROR_CODE.SUCCESS:
                print(f"Image grab failed at frame {frame_index}: {err}")
                return 1

            err = zed.retrieve_image(left_image, sl.VIEW.LEFT)
            if err > sl.ERROR_CODE.SUCCESS:
                print(f"Retrieve left image failed at frame {frame_index}: {err}")
                return 1

            err = zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            if err > sl.ERROR_CODE.SUCCESS:
                print(f"Retrieve right image failed at frame {frame_index}: {err}")
                return 1

            err = zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
            if err > sl.ERROR_CODE.SUCCESS:
                print(f"Retrieve depth image failed at frame {frame_index}: {err}")
                return 1

            err = zed.retrieve_measure(depth_measure, sl.MEASURE.DEPTH)
            if err > sl.ERROR_CODE.SUCCESS:
                print(f"Retrieve depth measure failed at frame {frame_index}: {err}")
                return 1

            left_frame = to_bgr(left_image)
            right_frame = to_bgr(right_image)
            depth_frame = to_bgr(depth_image)
            depth_metric_frame = depth_meters_to_png_mm(depth_measure.get_data())
            split_frame = build_split_frame(left_frame, depth_frame)

            left_writer.write(left_frame)
            right_writer.write(right_frame)
            depth_writer.write(depth_frame)
            split_writer.write(split_frame)

            if not cv2.imwrite(
                (left_frames_dir / f"left_{frame_index:06d}.bmp").as_posix(),
                left_frame,
            ):
                print(f"Write left frame failed at frame {frame_index}")
                return 1

            if not cv2.imwrite(
                (right_frames_dir / f"right_{frame_index:06d}.bmp").as_posix(),
                right_frame,
            ):
                print(f"Write right frame failed at frame {frame_index}")
                return 1

            if not cv2.imwrite(
                (depth_frames_dir / f"depth_2d_{frame_index:06d}.png").as_posix(),
                depth_frame,
            ):
                print(f"Write depth frame failed at frame {frame_index}")
                return 1

            if not cv2.imwrite(
                (depth_metric_frames_dir / f"depth_metric_mm_{frame_index:06d}.png").as_posix(),
                depth_metric_frame,
            ):
                print(f"Write depth metric frame failed at frame {frame_index}")
                return 1

            if not cv2.imwrite(
                (split_frames_dir / f"split_left_depth_{frame_index:06d}.png").as_posix(),
                split_frame,
            ):
                print(f"Write split frame failed at frame {frame_index}")
                return 1
            print(
                f"Saved frame {frame_index:06d}: "
                f"left_{frame_index:06d}.bmp, "
                f"right_{frame_index:06d}.bmp, "
                f"depth_2d_{frame_index:06d}.png, "
                f"depth_metric_mm_{frame_index:06d}.png, "
                f"split_left_depth_{frame_index:06d}.png"
            )

        info = zed.get_camera_information()
        calib = info.camera_configuration.calibration_parameters
        left = calib.left_cam
        right = calib.right_cam
        resolution = info.camera_configuration.resolution

        print("camera:")
        print(f"  model: {info.camera_model.name}")
        print(f"  serial_number: {info.serial_number}")
        print(f"  resolution: {resolution.width}x{resolution.height}")
        print(f"  depth_mode: {DEPTH_MODE.name}")
        print("intrinsics_rectified:")
        print(
            f"  left: fx={left.fx}, fy={left.fy}, cx={left.cx}, cy={left.cy}, "
            f"size={left.image_size.width}x{left.image_size.height}"
        )
        print(
            f"  right: fx={right.fx}, fy={right.fy}, cx={right.cx}, cy={right.cy}, "
            f"size={right.image_size.width}x{right.image_size.height}"
        )
        print("outputs:")
        print(f"  capture_dir: {capture_dir}")
        print(f"  depth_metric_mm_frames: {depth_metric_frames_dir}")
    finally:
        if left_writer is not None:
            left_writer.release()
        if right_writer is not None:
            right_writer.release()
        if depth_writer is not None:
            depth_writer.release()
        if split_writer is not None:
            split_writer.release()
        zed.close()

    return 0


def _self_check() -> None:
    with TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        first = next_capture_dir(output_dir, "zed2i_video")
        assert first.name == "zed2i_video_001"
        first.mkdir()
        second = next_capture_dir(output_dir, "zed2i_video")
        assert second.name == "zed2i_video_002"
    left = np.zeros((10, 20, 3), dtype=np.uint8)
    depth = np.zeros((10, 20, 3), dtype=np.uint8)
    split = build_split_frame(left, depth)
    assert split.shape == (10, 40, 3)
    metric = depth_meters_to_png_mm(
        np.array([[np.nan, 0.0, 1.234], [65.536, 4.0, np.inf]], dtype=np.float32)
    )
    assert metric.dtype == np.uint16
    assert metric.tolist() == [[0, 0, 1234], [65535, 4000, 0]]


if __name__ == "__main__":
    _self_check()
    raise SystemExit(main())
