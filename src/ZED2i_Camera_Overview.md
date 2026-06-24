USE WINDOWS

## Setup

1. Install matching ZED SDK for your CUDA version.
2. Release page: https://www.stereolabs.com/en-fr/developers/release#CUDA13
3. Make venv.
4. Run `get_python_api.py`.
5. Main Python docs: https://www.stereolabs.com/docs/api/python/index.html

## ZED 2i Overview

### Video modes for ZED 2i

These are official ZED 2/2i/Mini modes from Stereolabs camera controls page.

| Mode | Per-eye resolution | Side-by-side stream | FPS | FOV |
| --- | --- | --- | --- | --- |
| `HD2K` | `2208x1242` | `4416x1242` | `15` | Wide |
| `HD1080` | `1920x1080` | `3840x1080` | `30, 15` | Wide |
| `HD720` | `1280x720` | `2560x720` | `60, 30, 15` | Extra Wide |
| `VGA` | `672x376` | `1344x376` | `100, 60, 30, 15` | Extra Wide |

Notes:
- `sl.RESOLUTION` enum contains modes for other cameras too, like `HD1200`, `SVGA`, `HD4K`, `XVGA`, `TXGA`. Do not assume those belong to ZED 2i.
- `AUTO` usually resolves to `HD720` on non-ZED-X cameras.

### Depth modes

SDK exposes these `sl.DEPTH_MODE` values:

| Mode | What it means | Tradeoff |
| --- | --- | --- |
| `NONE` | No depth map. Rectified stereo images only. | Fastest, no depth |
| `PERFORMANCE` | Classic stereo mode, speed-first | Less detail than heavier modes |
| `QUALITY` | Classic stereo mode for harder low-texture areas | Slower than `PERFORMANCE` |
| `ULTRA` | Classic stereo mode favoring edges/sharpness | More GPU/memory |
| `NEURAL_LIGHT` | AI depth, light model | Fastest neural, least detail |
| `NEURAL` | AI depth, balanced model | Good default |
| `NEURAL_PLUS` | AI depth, heaviest model | Best detail/stability, slowest |
| `CUSTOM` | You provide depth/disparity yourself | No internal depth |

For ZED 2i depth range, official docs say:
- Wide lens: max `0.3m` to `20m`, ideal `0.3m` to `12m`
- Narrow lens: max `1.5m` to `35m`, ideal `1.5m` to `20m`

### Neural engines

Depth AI engines exposed in `sl.AI_MODELS`:

| Depth mode | AI model | Good | Bad |
| --- | --- | --- | --- |
| `NEURAL_LIGHT` | `NEURAL_LIGHT_DEPTH` | Lowest compute, good for multi-camera | Least detail, shortest ideal range |
| `NEURAL` | `NEURAL_DEPTH` | Balanced speed/detail | Not best at either extreme |
| `NEURAL_PLUS` | `NEURAL_PLUS_DEPTH` | Best detail, range, robustness | Highest GPU load, slowest |

Useful SDK helpers:
- `sl.check_ai_model_status(model)`
- `sl.download_ai_model(model)`
- `sl.optimize_ai_model(model)`

### Filters and thresholds

Main runtime depth filters/toggles:

| Knob | Where | What it does |
| --- | --- | --- |
| `enable_depth` | `sl.RuntimeParameters` | Turn depth on/off per `grab()` |
| `confidence_threshold` | `sl.RuntimeParameters` | Lower = stricter reject of weak depth points |
| `texture_confidence_threshold` | `sl.RuntimeParameters` | Lower = stricter reject in flat/low-texture areas |
| `remove_saturated_areas` | `sl.RuntimeParameters` | Remove bright saturated areas from depth |
| `enable_fill_mode` | `sl.RuntimeParameters` | Fill depth holes; overrides confidence/texture/saturated filters |
| `depth_stabilization` | `sl.InitParameters` | Temporal smoothing. `0` disables |
| `depth_minimum_distance` | `sl.InitParameters` | Clamp nearest allowed depth |
| `depth_maximum_distance` | `sl.InitParameters` | Clamp farthest allowed depth |

Python example:

```python
runtime_params = sl.RuntimeParameters()
runtime_params.enable_depth = True
runtime_params.confidence_threshold = 50
runtime_params.texture_confidence_threshold = 50
runtime_params.remove_saturated_areas = False
runtime_params.enable_fill_mode = False

init_params = sl.InitParameters()
init_params.depth_stabilization = 0
init_params.depth_minimum_distance = 0.3
init_params.depth_maximum_distance = 12
```

Rules:
- Lower threshold value = stronger filtering.
- `enable_fill_mode = True` overrides `confidence_threshold`, `texture_confidence_threshold`, and `remove_saturated_areas`.
- `depth_stabilization != 0` adds compute and uses tracking in background.

### Intrinsics/extrinsics and resolution

Calibration is resolution-dependent.

Use:
- `camera_information.camera_configuration.calibration_parameters`
  - rectified / undistorted intrinsics + stereo transform
- `camera_information.camera_configuration.calibration_parameters_raw`
  - raw / distorted intrinsics + stereo transform

Per-eye intrinsics:
- `fx`, `fy`
- `cx`, `cy`
- `disto`
- `h_fov`, `v_fov`, `d_fov`
- `image_size`

Stereo extrinsics:
- `stereo_transform`
- `get_camera_baseline()`

Important:
- `zed.get_camera_information()` returns calibration for current output resolution.
- `zed.get_camera_information(sl.get_resolution(sl.RESOLUTION.HD720))` returns calibration scaled for that resolution.
- If self-calibration stays enabled, startup can slightly refine calibration. To keep factory values only: `init_params.camera_disable_self_calib = True`

Python example:

```python
info_720 = zed.get_camera_information(sl.get_resolution(sl.RESOLUTION.HD720))
rectified = info_720.camera_configuration.calibration_parameters
raw = info_720.camera_configuration.calibration_parameters_raw

print(rectified.left_cam.fx, rectified.left_cam.fy)
print(rectified.stereo_transform.get_translation().get())
print(raw.left_cam.disto)
```


## Sources

- Local SDK stub: `venv_zed2i/Lib/site-packages/pyzed/sl.pyi`
- Camera controls: https://www.stereolabs.com/docs/video/camera-controls
- Depth modes: https://www.stereolabs.com/docs/depth-sensing/depth-modes
- Depth settings: https://www.stereolabs.com/docs/depth-sensing/depth-settings
- Camera calibration: https://www.stereolabs.com/docs/video/camera-calibration
