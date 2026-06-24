import enum

import pyzed.sl as sl


ZED2I_VIDEO_MODES = [
    {
        "mode": sl.RESOLUTION.HD2K,
        "per_eye": "2208x1242",
        "side_by_side": "4416x1242",
        "fps": [15],
        "fov": "Wide",
    },
    {
        "mode": sl.RESOLUTION.HD1080,
        "per_eye": "1920x1080",
        "side_by_side": "3840x1080",
        "fps": [30, 15],
        "fov": "Wide",
    },
    {
        "mode": sl.RESOLUTION.HD720,
        "per_eye": "1280x720",
        "side_by_side": "2560x720",
        "fps": [60, 30, 15],
        "fov": "Extra Wide",
    },
    {
        "mode": sl.RESOLUTION.VGA,
        "per_eye": "672x376",
        "side_by_side": "1344x376",
        "fps": [100, 60, 30, 15],
        "fov": "Extra Wide",
    },
]

DEPTH_MODE_NOTES = {
    sl.DEPTH_MODE.NONE: "No depth. Rectified stereo images only.",
    sl.DEPTH_MODE.PERFORMANCE: "Fastest classic stereo mode.",
    sl.DEPTH_MODE.QUALITY: "Classic stereo mode for harder low-texture areas.",
    sl.DEPTH_MODE.ULTRA: "Classic stereo mode favoring edges/sharpness. More GPU.",
    sl.DEPTH_MODE.NEURAL_LIGHT: "AI depth. Fastest neural mode. Best for speed/multi-camera.",
    sl.DEPTH_MODE.NEURAL: "AI depth. Balanced default neural mode.",
    sl.DEPTH_MODE.NEURAL_PLUS: "AI depth. Most detail/robustness. Most GPU and slowest.",
    sl.DEPTH_MODE.CUSTOM: "No internal depth. You ingest your own depth/disparity.",
}

DEPTH_AI_ENGINES = [
    (
        sl.DEPTH_MODE.NEURAL_LIGHT,
        sl.AI_MODELS.NEURAL_LIGHT_DEPTH,
        "fastest, least detail, lowest GPU",
    ),
    (
        sl.DEPTH_MODE.NEURAL,
        sl.AI_MODELS.NEURAL_DEPTH,
        "balanced speed/detail",
    ),
    (
        sl.DEPTH_MODE.NEURAL_PLUS,
        sl.AI_MODELS.NEURAL_PLUS_DEPTH,
        "best detail/range/stability, highest GPU",
    ),
]

SETTINGS = [
    sl.VIDEO_SETTINGS.BRIGHTNESS,
    sl.VIDEO_SETTINGS.CONTRAST,
    sl.VIDEO_SETTINGS.HUE,
    sl.VIDEO_SETTINGS.SATURATION,
    sl.VIDEO_SETTINGS.SHARPNESS,
    sl.VIDEO_SETTINGS.GAMMA,
    sl.VIDEO_SETTINGS.GAIN,
    sl.VIDEO_SETTINGS.EXPOSURE,
    sl.VIDEO_SETTINGS.AEC_AGC,
    sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE,
    sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO,
    sl.VIDEO_SETTINGS.LED_STATUS,
    sl.VIDEO_SETTINGS.EXPOSURE_TIME,
    sl.VIDEO_SETTINGS.ANALOG_GAIN,
    sl.VIDEO_SETTINGS.DIGITAL_GAIN,
    sl.VIDEO_SETTINGS.EXPOSURE_COMPENSATION,
    sl.VIDEO_SETTINGS.DENOISING,
    sl.VIDEO_SETTINGS.SCENE_ILLUMINANCE,
    sl.VIDEO_SETTINGS.AE_ANTIBANDING,
]

RANGE_SETTINGS = [
    sl.VIDEO_SETTINGS.AUTO_EXPOSURE_TIME_RANGE,
    sl.VIDEO_SETTINGS.AUTO_ANALOG_GAIN_RANGE,
    sl.VIDEO_SETTINGS.AUTO_DIGITAL_GAIN_RANGE,
]

X_FAMILY_MODELS = {
    sl.MODEL.ZED_X,
    sl.MODEL.ZED_XM,
    sl.MODEL.ZED_X_NANO,
    sl.MODEL.ZED_X_HDR,
    sl.MODEL.ZED_X_HDR_MINI,
    sl.MODEL.ZED_X_HDR_MAX,
    sl.MODEL.VIRTUAL_ZED_X,
    sl.MODEL.ZED_XONE_GS,
    sl.MODEL.ZED_XONE_UHD,
    sl.MODEL.ZED_XONE_HDR,
}

X_ONLY_SETTINGS = {
    sl.VIDEO_SETTINGS.EXPOSURE_TIME,
    sl.VIDEO_SETTINGS.ANALOG_GAIN,
    sl.VIDEO_SETTINGS.DIGITAL_GAIN,
    sl.VIDEO_SETTINGS.EXPOSURE_COMPENSATION,
    sl.VIDEO_SETTINGS.DENOISING,
    sl.VIDEO_SETTINGS.SCENE_ILLUMINANCE,
    sl.VIDEO_SETTINGS.AE_ANTIBANDING,
    sl.VIDEO_SETTINGS.AUTO_EXPOSURE_TIME_RANGE,
    sl.VIDEO_SETTINGS.AUTO_ANALOG_GAIN_RANGE,
    sl.VIDEO_SETTINGS.AUTO_DIGITAL_GAIN_RANGE,
}


def _section(title):
    print(f"\n=== {title} ===")


def _camera_params_summary(camera_params):
    return {
        "fx": camera_params.fx,
        "fy": camera_params.fy,
        "cx": camera_params.cx,
        "cy": camera_params.cy,
        "h_fov_deg": camera_params.h_fov,
        "v_fov_deg": camera_params.v_fov,
        "d_fov_deg": camera_params.d_fov,
        "focal_length_mm": camera_params.focal_length_metric,
        "image_size": _plain(camera_params.image_size),
        "distortion_model": _plain(camera_params.lens_distortion_model),
        "disto": _plain(camera_params.disto),
    }


def _calibration_summary(calibration):
    return {
        "baseline": calibration.get_camera_baseline(),
        "stereo_transform": _plain(calibration.stereo_transform),
        "left_cam": _camera_params_summary(calibration.left_cam),
        "right_cam": _camera_params_summary(calibration.right_cam),
    }


def _unsupported_reason(camera_info, setting):
    model = camera_info.camera_model
    firmware = camera_info.camera_configuration.firmware_version

    if setting in X_ONLY_SETTINGS and model not in X_FAMILY_MODELS:
        return f"unsupported on {model.name}; ZED X family only"
    if setting == sl.VIDEO_SETTINGS.LED_STATUS and firmware < 1523:
        return f"unsupported on firmware {firmware}; need >= 1523"
    return None


def _public_fields(obj):
    fields = []
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception as exc:  # pragma: no cover - camera SDK surface
            fields.append((name, f"<error: {exc}>"))
            continue
        if callable(value):
            continue
        fields.append((name, value))
    return fields


def _plain(value):
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, sl.Resolution):
        return {"width": value.width, "height": value.height, "area": value.area()}
    if isinstance(value, sl.Translation):
        return value.get().tolist()
    if isinstance(value, sl.Orientation):
        return value.get().tolist()
    if isinstance(value, sl.Transform):
        return {
            "translation": value.get_translation().get().tolist(),
            "euler_deg": value.get_euler_angles(False).tolist(),
            "matrix": value.m.tolist(),
        }
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _dump(name, value, indent=0, seen=None):
    seen = seen or set()
    prefix = "  " * indent

    if isinstance(value, (str, int, float, bool)) or value is None:
        print(f"{prefix}{name}: {value}")
        return

    plain = _plain(value)
    if plain is not value:
        print(f"{prefix}{name}: {plain}")
        return

    object_id = id(value)
    if object_id in seen:
        print(f"{prefix}{name}: <already shown>")
        return
    seen.add(object_id)

    print(f"{prefix}{name}:")
    for field_name, field_value in _public_fields(value):
        _dump(field_name, field_value, indent + 1, seen)


def _print_devices():
    _section("Connected Devices")
    devices = sl.Camera.get_device_list()
    if not devices:
        print("No devices detected.")
        return
    for index, device in enumerate(devices):
        _dump(f"device[{index}]", device)


def _print_camera_settings(zed, camera_info):
    _section("Current Camera Settings")
    for setting in SETTINGS:
        reason = _unsupported_reason(camera_info, setting)
        if reason:
            print(f"{setting.name}: {reason}")
            continue
        err, value = zed.get_camera_settings(setting)
        text = value if err == sl.ERROR_CODE.SUCCESS else err
        if err == sl.ERROR_CODE.INVALID_FUNCTION_PARAMETERS:
            text = f"unsupported by current camera/runtime ({err})"
        elif err == sl.ERROR_CODE.FAILURE:
            text = f"query failed ({err})"
        print(f"{setting.name}: {text}")
    for setting in RANGE_SETTINGS:
        reason = _unsupported_reason(camera_info, setting)
        if reason:
            print(f"{setting.name}: {reason}")
            continue
        err, min_value, max_value = zed.get_camera_settings_range(setting)
        if err == sl.ERROR_CODE.SUCCESS:
            text = f"{min_value} .. {max_value}"
        elif err == sl.ERROR_CODE.INVALID_FUNCTION_PARAMETERS:
            text = f"unsupported by current camera/runtime ({err})"
        else:
            text = err
        print(f"{setting.name}: {text}")


def _print_supported_video_modes(camera_info):
    _section("Supported Video Modes")
    model = camera_info.camera_model
    if model != sl.MODEL.ZED2i:
        print(f"Built-in table below is for ZED2i. Current model: {model.name}")
    for item in ZED2I_VIDEO_MODES:
        fps_text = ", ".join(str(fps) for fps in item["fps"])
        print(
            f"{item['mode'].name}: per-eye {item['per_eye']}, side-by-side {item['side_by_side']}, "
            f"fps [{fps_text}], fov {item['fov']}"
        )


def _print_depth_modes():
    _section("Depth Modes")
    for mode in sl.DEPTH_MODE:
        if mode == sl.DEPTH_MODE.LAST:
            continue
        print(f"{mode.name}: {DEPTH_MODE_NOTES.get(mode, 'See SDK docs.')}")


def _print_depth_ai_engines():
    _section("Depth AI Engines")
    for depth_mode, ai_model, tradeoff in DEPTH_AI_ENGINES:
        print(f"{depth_mode.name}: engine {ai_model.name} -> {tradeoff}")
        try:
            print(f"  status: {sl.check_ai_model_status(ai_model)}")
        except Exception as exc:  # pragma: no cover - runtime SDK/GPU dependent
            print(f"  status: <error: {exc}>")
    print("download/optimize helpers: sl.download_ai_model(model), sl.optimize_ai_model(model)")


def _print_depth_filter_knobs(zed):
    _section("Depth Filters And Toggles")
    runtime_params = zed.get_runtime_parameters()
    init_params = zed.get_init_parameters()
    print(f"enable_depth: {runtime_params.enable_depth}  # False = no depth this grab")
    print(f"confidence_threshold: {runtime_params.confidence_threshold}  # lower = stricter filter")
    print(
        f"texture_confidence_threshold: {runtime_params.texture_confidence_threshold}  "
        "# lower = stricter filter in flat/low-texture areas"
    )
    print(
        f"remove_saturated_areas: {runtime_params.remove_saturated_areas}  "
        "# False disables saturated-area filter"
    )
    print(
        f"enable_fill_mode: {runtime_params.enable_fill_mode}  "
        "# True fills holes, overrides confidence/texture/saturated filters"
    )
    print(
        f"depth_stabilization: {init_params.depth_stabilization}  "
        "# 0 off, higher = more temporal smoothing, more compute"
    )
    print("how-to:")
    print("  runtime_params.confidence_threshold = 50")
    print("  runtime_params.texture_confidence_threshold = 50")
    print("  runtime_params.remove_saturated_areas = False")
    print("  runtime_params.enable_fill_mode = True")
    print("  runtime_params.enable_depth = False")
    print("  init_params.depth_stabilization = 0")


def _print_calibration_by_resolution(zed):
    _section("Calibration By Resolution")
    for item in ZED2I_VIDEO_MODES:
        resolution = sl.get_resolution(item["mode"])
        info = zed.get_camera_information(resolution)
        print(f"{item['mode'].name}:")
        print(f"  scaled_resolution: {_plain(info.camera_configuration.resolution)}")
        print(
            f"  rectified: {_calibration_summary(info.camera_configuration.calibration_parameters)}"
        )
        print(
            f"  raw: {_calibration_summary(info.camera_configuration.calibration_parameters_raw)}"
        )


def _self_check():
    fields = {name for name, _ in _public_fields(sl.InitParameters())}
    assert "camera_resolution" in fields
    assert _plain([1, 2, 3]) == [1, 2, 3]
    assert ZED2I_VIDEO_MODES[0]["mode"] == sl.RESOLUTION.HD2K
    assert DEPTH_MODE_NOTES[sl.DEPTH_MODE.NEURAL].startswith("AI depth")
    reason = _unsupported_reason(
        type(
            "CameraInfoStub",
            (),
            {
                "camera_model": sl.MODEL.ZED2i,
                "camera_configuration": type("Cfg", (), {"firmware_version": 1000})(),
            },
        )(),
        sl.VIDEO_SETTINGS.EXPOSURE_TIME,
    )


def main():
    _self_check()
    _print_devices()

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.sdk_verbose = 0
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # ponytail: info dump only, turn depth back on only if you need depth outputs too.

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Camera open failed: {err}")
        return 1

    try:
        camera_info = zed.get_camera_information()

        _section("Requested Init Parameters")
        _dump("init_params", init_params)

        _section("Effective Init Parameters")
        _dump("init_params", zed.get_init_parameters())

        _section("Runtime Parameters")
        _dump("runtime_params", zed.get_runtime_parameters())

        _section("Camera Information")
        _dump("camera_information", camera_info)

        _print_supported_video_modes(camera_info)
        _print_depth_modes()
        _print_depth_ai_engines()
        _print_depth_filter_knobs(zed)
        _print_calibration_by_resolution(zed)
        _print_camera_settings(zed, camera_info)
    finally:
        zed.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
