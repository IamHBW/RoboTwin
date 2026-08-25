from __future__ import annotations

import importlib
import inspect
import json
import os
import pickle
import csv
import subprocess
import sys
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MODEL_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)


def _resolve_env(value: Any, default: Any = "") -> Any:
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return default
    if text.startswith("${") and text.endswith("}"):
        return os.environ.get(text[2:-1], default)
    if text.startswith("env:"):
        return os.environ.get(text[4:], default)
    if text.startswith("$") and len(text) > 1:
        return os.environ.get(text[1:], default)
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    value = _resolve_env(value, default)
    return default if value in (None, "") else int(value)


def _as_float(value: Any, default: float) -> float:
    value = _resolve_env(value, default)
    return default if value in (None, "") else float(value)


def _as_image_size(value: Any) -> tuple[int, int]:
    value = _resolve_env(value, [240, 320])
    if isinstance(value, str):
        parts = [int(part) for part in value.replace("x", ",").split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"image_size must have two integers, got {value!r}")
        return parts[0], parts[1]
    if len(value) != 2:
        raise ValueError(f"image_size must have two integers, got {value!r}")
    return int(value[0]), int(value[1])


def _as_camera_list(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    value = _resolve_env(value, ",".join(default))
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    return tuple(items or default)


@dataclass
class WAMConfig:
    model_type: str
    backend: str
    rgbdwam_root: str
    wam_checkpoint: str
    v2w_checkpoint: str
    vae_checkpoint: str
    inference_module: str
    inference_factory: str
    server_url: str
    server_timeout_s: float
    camera: str
    view_layout: str
    view_cameras: tuple[str, ...]
    image_size: tuple[int, int]
    obs_video_horizon: int
    action_horizon: int
    action_per_frame: int
    video_frame_stride: int
    actions_per_eval: int
    depth_unit_scale: float
    depth_min_m: float
    depth_max_m: float
    normalize_depth: bool
    clip_gripper: bool
    learn_row: bool
    trace_enabled: bool
    trace_dir: str

    @classmethod
    def from_args(cls, usr_args: dict[str, Any]) -> "WAMConfig":
        model_type = str(_resolve_env(usr_args.get("model_type"), os.environ.get("WAM_MODEL_TYPE", "depth"))).lower()
        if model_type not in {"rgb", "depth"}:
            raise ValueError(f"model_type must be 'rgb' or 'depth', got {model_type!r}")

        root = str(_resolve_env(usr_args.get("rgbdwam_root"), os.environ.get("RGBDWAM_ROOT", "")))
        if not root:
            root = "/mnt/data/users/tianyu/workspace/code/rgbdwam"
        camera = str(_resolve_env(usr_args.get("camera"), os.environ.get("WAM_CAMERA", "front_camera")))
        view_layout = str(_resolve_env(usr_args.get("view_layout"), os.environ.get("WAM_VIEW_LAYOUT", "single"))).lower()
        if view_layout not in {"single", "tshape"}:
            raise ValueError(f"view_layout must be 'single' or 'tshape', got {view_layout!r}")
        default_view_cameras = (camera,) if view_layout == "single" else ("head_camera", "left_camera", "right_camera")
        view_cameras = _as_camera_list(usr_args.get("view_cameras", os.environ.get("WAM_VIEW_CAMERAS", "")), default_view_cameras)
        expected_views = 1 if view_layout == "single" else 3
        if len(view_cameras) != expected_views:
            raise ValueError(f"view_layout={view_layout!r} expects {expected_views} cameras, got {view_cameras}")
        image_size_default = [240, 320] if view_layout == "single" else [256, 320]
        raw_image_size = os.environ.get("WAM_IMAGE_SIZE", "") or usr_args.get("image_size", image_size_default)
        if view_layout == "tshape" and os.environ.get("WAM_IMAGE_SIZE", "") == "" and raw_image_size == [240, 320]:
            raw_image_size = image_size_default

        return cls(
            model_type=model_type,
            backend=str(_resolve_env(usr_args.get("backend"), os.environ.get("WAM_BACKEND", "local"))).lower(),
            rgbdwam_root=root,
            wam_checkpoint=str(_resolve_env(usr_args.get("wam_checkpoint"), os.environ.get("WAM_CHECKPOINT", ""))),
            v2w_checkpoint=str(_resolve_env(usr_args.get("v2w_checkpoint"), os.environ.get("WAM_V2W_CHECKPOINT", ""))),
            vae_checkpoint=str(_resolve_env(usr_args.get("vae_checkpoint"), os.environ.get("WAM_VAE_CHECKPOINT", ""))),
            inference_module=str(
                _resolve_env(usr_args.get("inference_module"), os.environ.get("WAM_INFERENCE_MODULE", ""))
            ),
            inference_factory=str(
                _resolve_env(usr_args.get("inference_factory"), os.environ.get("WAM_INFERENCE_FACTORY", ""))
            ),
            server_url=str(_resolve_env(usr_args.get("server_url"), os.environ.get("WAM_SERVER_URL", ""))),
            server_timeout_s=_as_float(usr_args.get("server_timeout_s"), 120.0),
            camera=camera,
            view_layout=view_layout,
            view_cameras=view_cameras,
            image_size=_as_image_size(raw_image_size),
            obs_video_horizon=_as_int(usr_args.get("obs_video_horizon"), 5),
            action_horizon=_as_int(usr_args.get("action_horizon"), 16),
            action_per_frame=_as_int(usr_args.get("action_per_frame"), 4),
            video_frame_stride=_as_int(usr_args.get("video_frame_stride"), 1),
            actions_per_eval=_as_int(usr_args.get("actions_per_eval"), 4),
            depth_unit_scale=_as_float(usr_args.get("depth_unit_scale"), 0.001),
            depth_min_m=_as_float(usr_args.get("depth_min_m"), 0.05),
            depth_max_m=_as_float(usr_args.get("depth_max_m"), 1.2),
            normalize_depth=_as_bool(_resolve_env(usr_args.get("normalize_depth"), True)),
            clip_gripper=_as_bool(_resolve_env(usr_args.get("clip_gripper"), True)),
            learn_row=_as_bool(
                _resolve_env(
                    usr_args.get("learn_row"),
                    os.environ.get("WAM_LEARN_ROW", os.environ.get("WAM_ROT6D_LEARN_ROW", False)),
                )
            ),
            trace_enabled=_as_bool(
                _resolve_env(usr_args.get("trace_enabled"), os.environ.get("WAM_TRACE_ENABLED", False))
            ),
            trace_dir=str(_resolve_env(usr_args.get("trace_dir"), os.environ.get("WAM_TRACE_DIR", ""))),
        )


def _install_rgbdwam_paths(root: str) -> None:
    root_path = Path(root).expanduser().resolve()
    candidates = [
        root_path,
        root_path / "mimic-video",
        root_path / "mimic-video" / "model",
    ]
    for candidate in reversed(candidates):
        text = str(candidate)
        if candidate.exists() and text not in sys.path:
            sys.path.insert(0, text)


class LocalWAMBackend:
    """Adapter around a user-provided WAM inference module.

    The module may expose one of:
      - build_model(config) / create_model(config) / load_model(config)
      - a class named by inference_factory
      - predict_action_chunk(payload), predict(payload), get_action(payload), or __call__(payload)
    """

    def __init__(self, config: WAMConfig, usr_args: dict[str, Any]):
        _install_rgbdwam_paths(config.rgbdwam_root)
        if not config.inference_module:
            raise RuntimeError(
                "WAM local backend requires inference_module. Set WAM_INFERENCE_MODULE or pass "
                "--inference_module. This adapter intentionally does not run dummy actions."
            )
        module = importlib.import_module(config.inference_module)
        self.model = self._build_model(module, config, usr_args)

    def _build_model(self, module: Any, config: WAMConfig, usr_args: dict[str, Any]) -> Any:
        if config.inference_factory:
            factory = getattr(module, config.inference_factory)
            return self._call_factory(factory, config, usr_args)
        for name in ("build_model", "create_model", "load_model", "get_model"):
            if hasattr(module, name):
                return self._call_factory(getattr(module, name), config, usr_args)
        if hasattr(module, "WAMInference"):
            return self._call_factory(module.WAMInference, config, usr_args)
        raise RuntimeError(
            f"{config.inference_module} must define build_model/create_model/load_model/get_model "
            "or WAMInference, or set WAM_INFERENCE_FACTORY."
        )

    @staticmethod
    def _call_factory(factory: Any, config: WAMConfig, usr_args: dict[str, Any]) -> Any:
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            return factory(config=config, usr_args=usr_args)

        params = list(signature.parameters.values())
        names = {param.name for param in params}
        has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)
        has_var_args = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params)
        positional = [
            param
            for param in params
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and param.default is inspect.Parameter.empty
        ]

        if has_var_kwargs or {"config", "usr_args"}.issubset(names):
            return factory(config=config, usr_args=usr_args)
        if has_var_args:
            return factory(config, usr_args)
        if "config" in names:
            return factory(config=config)
        if "usr_args" in names:
            return factory(usr_args=usr_args)
        if len(positional) >= 2:
            return factory(config, usr_args)
        if len(positional) == 1:
            name = positional[0].name
            return factory(usr_args if name in {"args", "usr_args", "cfg"} else config)
        return factory()

    def predict(self, payload: dict[str, Any]) -> Any:
        model = self.model
        for name in ("predict_action_chunk", "predict", "get_action"):
            if hasattr(model, name):
                return getattr(model, name)(payload)
        if callable(model):
            return model(payload)
        raise RuntimeError("WAM inference object must implement predict_action_chunk/predict/get_action or be callable.")

    def predict_action_chunk(self, payload: dict[str, Any]) -> np.ndarray:
        return _coerce_action_chunk(self.predict(payload))


class HttpWAMBackend:
    def __init__(self, config: WAMConfig):
        if not config.server_url:
            raise RuntimeError("WAM http backend requires server_url. Set WAM_SERVER_URL or pass --server_url.")
        self.url = config.server_url
        self.timeout_s = config.server_timeout_s

    def predict(self, payload: dict[str, Any]) -> Any:
        body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/python-pickle"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return pickle.loads(response.read())

    def predict_action_chunk(self, payload: dict[str, Any]) -> np.ndarray:
        return _coerce_action_chunk(self.predict(payload))


def _coerce_action_chunk(result: Any) -> np.ndarray:
    if isinstance(result, dict):
        for key in ("action_chunk", "actions", "action/lowdim_concat"):
            if key in result:
                result = result[key]
                break
    arr = np.asarray(result, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 20:
        raise ValueError(f"WAM action chunk must have shape [H, 20], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("WAM action chunk contains NaN or Inf.")
    return arr


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quat = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quat = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quat = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
    if quat[0] < 0:
        quat = -quat
    return quat.astype(np.float32)


def _quat_to_rot6(quat_wxyz: np.ndarray, *, learn_row: bool = False) -> np.ndarray:
    rot = _quat_wxyz_to_matrix(np.asarray(quat_wxyz, dtype=np.float32))
    if learn_row:
        return rot[0:2, :].reshape(6).astype(np.float32)
    return rot[:, 0:2].reshape(6).astype(np.float32)


def _rot6_to_quat(rot6: np.ndarray, *, learn_row: bool = False) -> np.ndarray:
    values = np.asarray(rot6, dtype=np.float32)
    if learn_row:
        rows = values.reshape(2, 3)
        x = _normalize(rows[0])
        y = rows[1] - x * float(np.dot(x, rows[1]))
    else:
        cols = values.reshape(3, 2)
        x = _normalize(cols[:, 0])
        y = cols[:, 1] - x * float(np.dot(x, cols[:, 1]))
    y = _normalize(y)
    if np.linalg.norm(x) < 1e-8 or np.linalg.norm(y) < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    z = np.cross(x, y)
    rot = np.stack([x, y, z], axis=0 if learn_row else 1)
    return _matrix_to_quat_wxyz(rot)


def _resize_rgb(rgb: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    h, w = image_size
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    image = image.resize((w, h), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    return (arr / 127.5 - 1.0).transpose(2, 0, 1).astype(np.float32)


def _resize_depth(depth_m: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    h, w = image_size
    image = Image.fromarray(np.asarray(depth_m, dtype=np.float32), mode="F")
    image = image.resize((w, h), resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def _single_video_to_uint8_rgb(video: np.ndarray) -> np.ndarray:
    arr = np.asarray(video, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"single pred_video must have shape [C,T,H,W], got {arr.shape}")
    if arr.shape[0] >= 3:
        frames = np.transpose(arr[:3], (1, 2, 3, 0))
        frames = (frames + 1.0) * 127.5
    elif arr.shape[0] == 1:
        gray = (arr[0] + 1.0) * 127.5
        frames = np.repeat(gray[..., None], 3, axis=-1)
    else:
        raise ValueError(f"pred_video channel dimension must be 1 or >=3, got {arr.shape[0]}")
    return np.clip(frames, 0, 255).astype(np.uint8)


def _resize_uint8_video(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), resample=Image.Resampling.BILINEAR)) for frame in frames],
        axis=0,
    ).astype(np.uint8)


def _multiview_video_to_tshape_uint8(video: np.ndarray) -> np.ndarray:
    arr = np.asarray(video, dtype=np.float32)
    if arr.ndim != 5 or arr.shape[0] != 3:
        raise ValueError(f"multiview pred_video must have shape [3,C,T,H,W], got {arr.shape}")
    head = _single_video_to_uint8_rgb(arr[0])
    left = _single_video_to_uint8_rgb(arr[1])
    right = _single_video_to_uint8_rgb(arr[2])
    wrist_size = (head.shape[1] // 2, head.shape[2] // 2)
    left = _resize_uint8_video(left, wrist_size)
    right = _resize_uint8_video(right, wrist_size)
    top = np.concatenate([left, right], axis=2)
    return np.concatenate([top, head], axis=1)


def _video_to_uint8_rgb(video: np.ndarray) -> np.ndarray:
    arr = np.asarray(video, dtype=np.float32)
    if arr.ndim == 6 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 5 and arr.shape[0] == 3 and arr.shape[1] in {1, 3}:
        return _multiview_video_to_tshape_uint8(arr)
    if arr.ndim != 4:
        raise ValueError(f"pred_video must have shape [C,T,H,W], [1,C,T,H,W], or [1,3,C,T,H,W], got {arr.shape}")
    return _single_video_to_uint8_rgb(arr)


def _write_mp4(path: Path, frames: np.ndarray, fps: int = 5) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be [T,H,W,3], got {frames.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert ffmpeg.stdin is not None
    ffmpeg.stdin.write(frames.tobytes())
    ffmpeg.stdin.close()
    ret = ffmpeg.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret} while writing {path}")


class WAMPolicy:
    def __init__(self, usr_args: dict[str, Any]):
        self.config = WAMConfig.from_args(usr_args)
        if self.config.backend == "local":
            self.backend = LocalWAMBackend(self.config, usr_args)
        elif self.config.backend == "http":
            self.backend = HttpWAMBackend(self.config)
        else:
            raise ValueError(f"Unsupported backend {self.config.backend!r}; use 'local' or 'http'.")

        video_cache_len = (self.config.obs_video_horizon - 1) * self.config.video_frame_stride + 1
        self.obs_cache: deque[dict[str, Any]] = deque(maxlen=max(self.config.obs_video_horizon, video_cache_len))
        self.action_cache: deque[dict[str, Any]] = deque()
        self.last_instruction = ""
        self.last_model_prompt = ""
        self.current_episode_index = -1
        trace_dir = self.config.trace_dir or str(Path(str(usr_args.get("eval_save_dir", "."))) / "wam_trace")
        self.trace_dir = Path(trace_dir).expanduser() if self.config.trace_enabled else None
        self.trace_csv_path = self.trace_dir / "actions_proprio.csv" if self.trace_dir is not None else None
        self.chunk_index = 0
        self.last_action_record: dict[str, Any] | None = None
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        self.obs_cache.clear()
        self.action_cache.clear()
        self.last_instruction = ""
        self.last_model_prompt = ""
        self.current_episode_index = -1
        self.chunk_index = 0
        self.last_action_record = None

    def update_obs(self, task_env: Any, observation: dict[str, Any]) -> None:
        obs = self.encode_obs(task_env, observation)
        self.obs_cache.append(obs)
        self.last_instruction = str(getattr(task_env, "get_instruction")())
        self.last_model_prompt = MODEL_PROMPT_TEMPLATE.format(task=self.last_instruction)
        self.current_episode_index = int(getattr(task_env, "test_num", -1))

    def encode_obs(self, task_env: Any, observation: dict[str, Any]) -> dict[str, Any]:
        camera_obs = self._get_camera_observations(task_env, observation)
        lowdim, base = self._read_lowdim(task_env, observation)
        return {
            "frame": self._preprocess_frame(camera_obs),
            "lowdim": lowdim,
            "base": base,
        }

    def _get_camera_observations(self, task_env: Any, observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
        obs = observation.get("observation", {})
        cameras = self.config.view_cameras if self.config.view_layout == "tshape" else (self.config.camera,)
        depth_by_camera = None

        out: dict[str, dict[str, Any]] = {}
        for camera in cameras:
            camera_obs = obs.get(camera)
            if camera_obs is None:
                raise KeyError(f"Camera {camera!r} is missing from observation keys={list(obs.keys())}")

            if self.config.model_type == "depth" and "depth" not in camera_obs:
                if depth_by_camera is None:
                    depth_by_camera = task_env.cameras.get_depth()
                if camera not in depth_by_camera:
                    raise KeyError(f"Depth fallback did not return camera {camera!r}")
                camera_obs = dict(camera_obs)
                camera_obs.update(depth_by_camera[camera])
            out[camera] = camera_obs

        return out

    def _preprocess_frame(self, camera_obs_by_name: dict[str, dict[str, Any]]) -> np.ndarray:
        frames = [
            self._preprocess_single_camera_frame(camera, camera_obs_by_name[camera])
            for camera in (self.config.view_cameras if self.config.view_layout == "tshape" else (self.config.camera,))
        ]
        if self.config.view_layout == "single":
            return frames[0]
        return np.stack(frames, axis=0).astype(np.float32)

    def _preprocess_single_camera_frame(self, camera: str, camera_obs: dict[str, Any]) -> np.ndarray:
        if self.config.model_type == "rgb":
            if "rgb" not in camera_obs:
                raise KeyError(f"RGB image missing for camera {camera!r}")
            return _resize_rgb(camera_obs["rgb"], self.config.image_size)

        if "depth" not in camera_obs:
            raise KeyError(f"Depth image missing for camera {camera!r}")
        depth_m = np.asarray(camera_obs["depth"], dtype=np.float32) * self.config.depth_unit_scale
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=self.config.depth_max_m, neginf=0.0)
        depth_m = np.clip(depth_m, self.config.depth_min_m, self.config.depth_max_m)
        if self.config.normalize_depth:
            depth_m = 2.0 * (depth_m - self.config.depth_min_m) / (self.config.depth_max_m - self.config.depth_min_m) - 1.0
        return _resize_depth(depth_m, self.config.image_size)[None].astype(np.float32)

    def _read_lowdim(self, task_env: Any, observation: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        endpose = observation.get("endpose") or {}
        left_pose = np.asarray(endpose.get("left_endpose", task_env.get_arm_pose("left")), dtype=np.float32)
        right_pose = np.asarray(endpose.get("right_endpose", task_env.get_arm_pose("right")), dtype=np.float32)
        left_gripper = np.asarray(endpose.get("left_gripper", task_env.robot.get_left_gripper_val()), dtype=np.float32)
        right_gripper = np.asarray(endpose.get("right_gripper", task_env.robot.get_right_gripper_val()), dtype=np.float32)

        left = np.concatenate(
            [left_pose[:3], _quat_to_rot6(left_pose[3:7], learn_row=self.config.learn_row), left_gripper.reshape(1)]
        )
        right = np.concatenate(
            [
                right_pose[:3],
                _quat_to_rot6(right_pose[3:7], learn_row=self.config.learn_row),
                right_gripper.reshape(1),
            ]
        )
        lowdim = np.concatenate([left, right]).astype(np.float32)[None]
        base = {
            "left_xyz": left_pose[:3].astype(np.float32),
            "right_xyz": right_pose[:3].astype(np.float32),
        }
        return lowdim, base

    def _video_tensor(self) -> np.ndarray:
        if not self.obs_cache:
            raise RuntimeError("Observation cache is empty.")
        frames = [item["frame"] for item in self.obs_cache]
        stride = max(1, self.config.video_frame_stride)
        needed = (self.config.obs_video_horizon - 1) * stride + 1
        while len(frames) < needed:
            frames.insert(0, frames[0])
        frames = frames[-needed:]
        selected = frames[::stride]
        if len(selected) != self.config.obs_video_horizon:
            selected = selected[-self.config.obs_video_horizon :]
        time_axis = 2 if selected[0].ndim == 4 else 1
        return np.stack(selected, axis=time_axis).astype(np.float32)

    def _current_lowdim(self) -> np.ndarray:
        return self.obs_cache[-1]["lowdim"].astype(np.float32)

    def _current_base(self) -> dict[str, np.ndarray]:
        return self.obs_cache[-1]["base"]

    def _payload(self) -> dict[str, Any]:
        video_key = "obs/workspace_rgb" if self.config.model_type == "rgb" else "obs/workspace_depth"
        video = self._video_tensor()
        lowdim = self._current_lowdim()
        return {
            "model_type": self.config.model_type,
            "instruction": self.last_instruction,
            "model_prompt": self.last_model_prompt,
            "camera": self.config.camera,
            "view_layout": self.config.view_layout,
            "view_cameras": self.config.view_cameras,
            "wam_checkpoint": self.config.wam_checkpoint,
            "v2w_checkpoint": self.config.v2w_checkpoint,
            "vae_checkpoint": self.config.vae_checkpoint,
            "trace_pred_video": self.config.trace_enabled,
            "batch": {
                video_key: video[None],
                "obs/lowdim_concat": lowdim[None],
            },
        }

    def _refill_actions(self) -> None:
        result = self.backend.predict(self._payload())
        action_chunk = _coerce_action_chunk(result)
        predicted_horizon = min(self.config.action_horizon, action_chunk.shape[0])
        execute_horizon = min(predicted_horizon, max(1, self.config.actions_per_eval))
        action_chunk = action_chunk[:execute_horizon]
        base = self._current_base()
        self.action_cache.clear()
        chunk_id = self.chunk_index
        self.chunk_index += 1
        if isinstance(result, dict):
            self._write_predicted_observation(result, chunk_id)
        server_elapsed_s = float(result.get("elapsed_s", np.nan)) if isinstance(result, dict) else np.nan
        for chunk_step, action20 in enumerate(action_chunk):
            self.action_cache.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_step": chunk_step,
                    "raw_action20": np.asarray(action20, dtype=np.float32),
                    "ee_action": self._action20_to_ee(action20, base),
                    "server_elapsed_s": server_elapsed_s,
                }
            )

    def _write_predicted_observation(self, result: dict[str, Any], chunk_id: int) -> None:
        if self.trace_dir is None or "pred_video" not in result:
            return
        pred_video = np.asarray(result["pred_video"], dtype=np.float32)
        frames = _video_to_uint8_rgb(pred_video)
        out_dir = self.trace_dir / "v2w_pred_chunks"
        episode_id = int(getattr(self, "current_episode_index", -1))
        episode_prefix = f"ep{episode_id:03d}" if episode_id >= 0 else "epunk"
        out_path = out_dir / f"{episode_prefix}_ch{chunk_id:03d}.mp4"
        _write_mp4(out_path, frames, fps=5)

        manifest_path = self.trace_dir / "v2w_pred_chunks.jsonl"
        record = {
            "episode_id": episode_id,
            "chunk_id": int(chunk_id),
            "output_video": str(out_path),
            "pred_video_shape": list(pred_video.shape),
            "num_frames": int(frames.shape[0]),
            "height": int(frames.shape[1]),
            "width": int(frames.shape[2]),
            "source": str(result.get("pred_video_source", "")),
            "num_sampling_step": result.get("pred_video_num_sampling_step"),
            "return_context_at_step": result.get("pred_video_return_context_at_step"),
            "num_conditional_frames": result.get("pred_video_num_conditional_frames"),
            "stop_after_step": result.get("stop_after_step"),
            "modality": result.get("modality", self.config.model_type),
            "view_layout": result.get("view_layout", self.config.view_layout),
            "view_cameras": list(result.get("view_cameras", self.config.view_cameras)),
            "language_conditioning": result.get("language_conditioning", ""),
            "server_elapsed_s": result.get("elapsed_s"),
        }
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _action20_to_ee(self, action20: np.ndarray, base: dict[str, np.ndarray]) -> np.ndarray:
        left = np.asarray(action20[:10], dtype=np.float32)
        right = np.asarray(action20[10:20], dtype=np.float32)

        left_xyz = base["left_xyz"] + left[:3]
        right_xyz = base["right_xyz"] + right[:3]
        left_quat = _rot6_to_quat(left[3:9], learn_row=self.config.learn_row)
        right_quat = _rot6_to_quat(right[3:9], learn_row=self.config.learn_row)
        left_gripper = float(left[9])
        right_gripper = float(right[9])
        if self.config.clip_gripper:
            left_gripper = float(np.clip(left_gripper, 0.0, 1.0))
            right_gripper = float(np.clip(right_gripper, 0.0, 1.0))

        return np.concatenate(
            [
                left_xyz.astype(np.float32),
                left_quat.astype(np.float32),
                np.asarray([left_gripper], dtype=np.float32),
                right_xyz.astype(np.float32),
                right_quat.astype(np.float32),
                np.asarray([right_gripper], dtype=np.float32),
            ]
        )

    def next_action(self) -> np.ndarray:
        if not self.action_cache:
            self._refill_actions()
        record = self.action_cache.popleft()
        self.last_action_record = record
        return np.asarray(record["ee_action"], dtype=np.float32)

    @staticmethod
    def _flatten(prefix: str, values: np.ndarray) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        return {f"{prefix}_{idx:02d}": float(value) for idx, value in enumerate(arr)}

    def record_executed_action(self, task_env: Any, action: np.ndarray) -> None:
        if self.trace_csv_path is None or self.last_action_record is None:
            return
        row: dict[str, Any] = {
            "episode_index": int(getattr(task_env, "test_num", -1)),
            "take_action_cnt": int(getattr(task_env, "take_action_cnt", -1)),
            "chunk_id": int(self.last_action_record["chunk_id"]),
            "chunk_step": int(self.last_action_record["chunk_step"]),
            "server_elapsed_s": float(self.last_action_record.get("server_elapsed_s", np.nan)),
            "eval_success": bool(getattr(task_env, "eval_success", False)),
        }
        row.update(self._flatten("pred_action20", self.last_action_record["raw_action20"]))
        row.update(self._flatten("executed_ee_action16", action))
        if self.obs_cache:
            row.update(self._flatten("proprio_lowdim20", self._current_lowdim()[0]))
        for arm in ("left", "right"):
            try:
                row.update(self._flatten(f"post_{arm}_endpose7", task_env.get_arm_pose(arm)))
            except Exception:
                pass
        write_header = not self.trace_csv_path.exists()
        self.trace_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def get_model(usr_args: dict[str, Any]) -> WAMPolicy:
    return WAMPolicy(usr_args)


def reset_model(model: WAMPolicy) -> None:
    model.reset()


def eval(TASK_ENV: Any, model: WAMPolicy, observation: dict[str, Any]) -> dict[str, Any]:
    model.update_obs(TASK_ENV, observation)
    latest_observation = observation
    for step_idx in range(max(1, model.config.actions_per_eval)):
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
        action = model.next_action()
        TASK_ENV.take_action(action, action_type="ee")
        if hasattr(model, "record_executed_action"):
            model.record_executed_action(TASK_ENV, action)
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
        if step_idx + 1 < model.config.actions_per_eval:
            latest_observation = TASK_ENV.get_obs()
            model.update_obs(TASK_ENV, latest_observation)
    return latest_observation
