from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

OUTPUT_DIR = Path(__file__).with_name("captures")
FILE_STEM = "zed2i"
WARMUP_GRABS = 5

CONFIDENCE_THRESHOLD = 95
TEXTURE_CONFIDENCE_THRESHOLD = 100
REMOVE_SATURATED_AREAS = False #removes overexposed areas from the depth map
ENABLE_FILL_MODE = False
DEPTH_STABILIZATION = 30

DEPTH_MINIMUM_DISTANCE_METERS = 2.0
DEPTH_MAXIMUM_DISTANCE_METERS = 4.0

#ROI for auto exposure and gain (AEC/AGC) in pixels, relative to the top-left corner of the image. 
AEC_AGC_ROI_X = 300
AEC_AGC_ROI_Y = 200
AEC_AGC_ROI_WIDTH = 2208-300-300
AEC_AGC_ROI_HEIGHT = 1242-200-200


def next_capture_stem(output_dir: Path, file_stem: str) -> str:
    index = 1
    while any(
        (output_dir / f"{file_stem}_{index:03d}{suffix}").exists()
        for suffix in ("_left.bmp", "_right.bmp", "_depth.pfm", "_pointcloud.ply")
    ):
        index += 1
    return f"{file_stem}_{index:03d}"


import cv2
import pyzed.sl as sl

DEPTH_MODE = sl.DEPTH_MODE.NEURAL_PLUS  # use sl.DEPTH_MODE.ULTRA for classic stereo, or sl.DEPTH_MODE.NEURAL_PLUS for neural network-based depth estimation


def main() -> int:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_stem = next_capture_stem(output_dir, FILE_STEM)

    left_path = output_dir / f"{capture_stem}_left.bmp"
    right_path = output_dir / f"{capture_stem}_right.bmp"
    depth_path = output_dir / f"{capture_stem}_depth.pfm"
    point_cloud_path = output_dir / f"{capture_stem}_pointcloud.ply"

    timings = {}

    t0 = perf_counter()
    init_params = sl.InitParameters()
    init_params.enable_image_enhancement = False
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.camera_fps = 15
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
    timings["filtering_setup_s"] = perf_counter() - t0

    zed = sl.Camera()
    err = zed.open(init_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print(f"Camera open failed: {err}")
        return 1

    #camera settings
    zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 4)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE, 0)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 4)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 4)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 5)

    # zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 50)
    # zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 50)

    zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)  # auto exposure/gain on
    zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)  # auto WB on
    # zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, 4200)  # manual WB, also turns auto off
    roi = sl.Rect(AEC_AGC_ROI_X, AEC_AGC_ROI_Y, AEC_AGC_ROI_WIDTH, AEC_AGC_ROI_HEIGHT)
    zed.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, roi, sl.SIDE.BOTH)


    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_map = sl.Mat()
    point_cloud = sl.Mat()

    try:
        for _ in range(WARMUP_GRABS):
            err = zed.grab(runtime_params)
            if err > sl.ERROR_CODE.SUCCESS:
                print(f"Warmup grab failed: {err}")
                return 1

        t0 = perf_counter()
        err = zed.grab(runtime_params)
        timings["image_acquisition_s"] = perf_counter() - t0
        if err > sl.ERROR_CODE.SUCCESS:
            print(f"Image grab failed: {err}")
            return 1

        timings["depth_calculation_s"] = timings["image_acquisition_s"]

        t0 = perf_counter()
        err = zed.retrieve_image(left_image, sl.VIEW.LEFT)
        if err > sl.ERROR_CODE.SUCCESS:
            print(f"Retrieve left image failed: {err}")
            return 1
        err = zed.retrieve_image(right_image, sl.VIEW.RIGHT)
        if err > sl.ERROR_CODE.SUCCESS:
            print(f"Retrieve right image failed: {err}")
            return 1
        timings["image_retrieval_s"] = perf_counter() - t0

        t0 = perf_counter()
        err = zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        if err > sl.ERROR_CODE.SUCCESS:
            print(f"Retrieve depth map failed: {err}")
            return 1
        err = depth_map.write(depth_path.as_posix())
        if err > sl.ERROR_CODE.SUCCESS:
            print(f"Write depth map failed: {err}")
            return 1
        timings["depth_export_s"] = perf_counter() - t0

        t0 = perf_counter()
        err = zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        if err > sl.ERROR_CODE.SUCCESS:
            print(f"Retrieve point cloud failed: {err}")
            return 1
        err = point_cloud.write(point_cloud_path.as_posix())
        if err > sl.ERROR_CODE.SUCCESS:
            print(f"Write point cloud failed: {err}")
            return 1
        timings["pointcloud_making_s"] = perf_counter() - t0

        t0 = perf_counter()
        if not cv2.imwrite(left_path.as_posix(), left_image.get_data()):
            print("Write left image failed")
            return 1
        if not cv2.imwrite(right_path.as_posix(), right_image.get_data()):
            print("Write right image failed")
            return 1
        timings["image_save_s"] = perf_counter() - t0

        info = zed.get_camera_information()
        calib = info.camera_configuration.calibration_parameters
        left = calib.left_cam
        right = calib.right_cam
        transform = calib.stereo_transform
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
        print("extrinsics_rectified:")
        print(f"  baseline: {calib.get_camera_baseline()}")
        print(f"  translation: {transform.get_translation().get().tolist()}")
        print(f"  rotation_rpy_deg: {transform.get_euler_angles(False).tolist()}")
        print("outputs:")
        print(f"  left_image: {left_path}")
        print(f"  right_image: {right_path}")
        print(f"  depth_map: {depth_path}")
        print(f"  point_cloud: {point_cloud_path}")
        print("filters:")
        print(f"  confidence_threshold: {CONFIDENCE_THRESHOLD}")
        print(f"  texture_confidence_threshold: {TEXTURE_CONFIDENCE_THRESHOLD}")
        print(f"  remove_saturated_areas: {REMOVE_SATURATED_AREAS}")
        print(f"  enable_fill_mode: {ENABLE_FILL_MODE}")
        print(f"  depth_stabilization: {DEPTH_STABILIZATION}")
        print("aec_agc_roi:")
        print(f"  x: {AEC_AGC_ROI_X}")
        print(f"  y: {AEC_AGC_ROI_Y}")
        print(f"  width: {AEC_AGC_ROI_WIDTH}")
        print(f"  height: {AEC_AGC_ROI_HEIGHT}")
        print("timings_s:")
        print(f"  filtering_setup: {timings['filtering_setup_s']:.6f}")
        print(f"  image_acquisition: {timings['image_acquisition_s']:.6f}")
        print(f"  depth_calculation: {timings['depth_calculation_s']:.6f}")
        print(f"  image_retrieval: {timings['image_retrieval_s']:.6f}")
        print(f"  depth_export: {timings['depth_export_s']:.6f}")
        print(f"  pointcloud_making: {timings['pointcloud_making_s']:.6f}")
        print(f"  image_save: {timings['image_save_s']:.6f}")
    finally:
        zed.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
