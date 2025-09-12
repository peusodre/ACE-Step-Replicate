"""
Replicate prediction interface for ACE-Step (extend + repaint only)
"""

import os
import tempfile
import inspect
from typing import Optional
from cog import BasePredictor, Input, Path

import torch
from acestep.pipeline_ace_step import ACEStepPipeline


class Predictor(BasePredictor):
    def setup(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        # pick a safe default dtype
        dtype = "bfloat16"
        try:
            if torch.backends.mps.is_available():
                dtype = "float32"
            elif torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                dtype = "float32"
            elif not torch.cuda.is_available():
                dtype = "float32"
        except Exception:
            dtype = "float32"

        self.pipeline = ACEStepPipeline(
            checkpoint_dir=None,
            device_id=0,
            dtype=dtype,
            torch_compile=True,
            cpu_offload=True,
            overlapped_decode=True,
        )
        self.pipeline.load_checkpoint()

    def _duration_sec(self, wav_path: str) -> float:
        try:
            import librosa
            return float(librosa.get_duration(path=wav_path))
        except Exception:
            import torchaudio
            info = torchaudio.info(wav_path)
            return info.num_frames / float(info.sample_rate)

    def _validate(self, task: str, input_audio_path: Optional[str],
                  repaint_start_time: float, repaint_end_time: float) -> None:
        if task not in ("extend", "repaint"):
            raise ValueError("task must be 'extend' or 'repaint'")
        if not input_audio_path:
            raise ValueError(f"Task '{task}' requires input_audio")
        if task == "repaint" and repaint_end_time <= repaint_start_time:
            raise ValueError("inpaint_end_time must be greater than inpaint_start_time")

    def predict(
        self,
        # ---- task (only these two) ------------------------------------------
        task: str = Input(choices=["extend", "repaint"], default="extend"),

        # ---- text conditions -------------------------------------------------
        prompt: str = Input(default="Electronic dance music, upbeat, energetic"),
        lyrics: str = Input(
            default="",
            description="Optional lyrics. Tags like [verse], [chorus], [bridge] are supported."
        ),

        # ---- audio IO --------------------------------------------------------
        input_audio: Path = Input(description="Source audio", default=None),

        # ---- repaint controls (active when task='repaint') -------------------
        inpaint_start_time: float = Input(default=0.0, ge=0.0),
        inpaint_end_time: float = Input(default=0.0, ge=0.0),
        repaint_strength: float = Input(
            default=0.5, ge=0.0, le=1.0,
            description="Higher = stronger change in the repainted window"
        ),

        # ---- extend controls (active when task='extend') ---------------------
        target_total_seconds: float = Input(
            default=60.0, ge=5.0, le=240.0,
            description="Total desired duration after extension (right-only extend)."
        ),
        seam_seconds: float = Input(default=0.75, ge=0.1, le=5.0),
        extend_bootstrap_edge_sec: float = Input(
            default=2.0, ge=0.5, le=10.0,
            description="Edge context used for tiling the tail seed."
        ),

        # ---- diffusion / guidance -------------------------------------------
        infer_steps: int = Input(default=60, ge=20, le=120),
        scheduler_type: str = Input(default="euler", choices=["euler", "heun", "pingpong"]),
        cfg_type: str = Input(default="apg", choices=["apg", "cfg", "cfg_star"]),
        guidance_scale: float = Input(default=12.0, ge=1.0, le=30.0),
        omega_scale: float = Input(default=8.0, ge=1.0, le=20.0),
        guidance_interval_decay: float = Input(default=0.9, ge=0.0, le=5.0),
        min_guidance_scale: float = Input(default=3.0, ge=0.0, le=30.0),

        # ---- seeds -----------------------------------------------------------
        seed: int = Input(default=None),
        variation_seed: int = Input(default=None),
        variation_strength: float = Input(
            default=0.0, ge=0.0, le=1.0,
            description="If >0 (and not repaint), used as retake variance."
        ),

        # ---- LoRA / output ---------------------------------------------------
        lora_name_or_path: str = Input(default="none"),
        lora_weight: float = Input(default=1.0, ge=0.0, le=2.0),
        output_format: str = Input(default="wav", choices=["wav", "mp3", "ogg"]),
        sample_rate: int = Input(default=48000, choices=[44100, 48000]),
    ) -> Path:
        # temp workspace + materialize input
        temp_dir = tempfile.mkdtemp()
        input_audio_path = None
        if input_audio is not None:
            input_audio_path = os.path.join(temp_dir, "input_audio.wav")
            with open(input_audio_path, "wb") as f, open(input_audio, "rb") as fi:
                f.write(fi.read())

        # validate
        self._validate(task, input_audio_path, inpaint_start_time, inpaint_end_time)

        # seeds
        manual_seeds = [seed] if seed is not None else None
        retake_seeds = [variation_seed] if variation_seed is not None else manual_seeds

        # build repaint window
        if task == "extend":
            src_len = self._duration_sec(input_audio_path)
            # right-only extend: repaint_end = desired TOTAL seconds
            repaint_start = 0.0          # ignored by the pipeline in extend mode
            repaint_end = float(target_total_seconds)
            # light defaults that work well for extend
            guidance_scale_eff = max(guidance_scale, 12.0)
            min_cfg_eff = max(min_guidance_scale, 6.0)
        else:
            repaint_start = float(inpaint_start_time)
            repaint_end = float(inpaint_end_time)
            guidance_scale_eff = guidance_scale
            min_cfg_eff = min_guidance_scale

        # choose retake variance
        if task == "repaint":
            retake_variance = float(repaint_strength)
        else:
            retake_variance = float(variation_strength) if variation_strength > 0 else 0.3

        # call the slim pipeline
        kwargs = dict(
            format=output_format,
            save_path=temp_dir,
            sample_rate=int(sample_rate),

            task=task,
            src_audio_path=input_audio_path,
            repaint_start=repaint_start,
            repaint_end=repaint_end,

            prompt=prompt,
            lyrics=lyrics,

            infer_step=int(infer_steps),
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            guidance_scale=float(guidance_scale_eff),
            omega_scale=float(omega_scale),
            guidance_interval_decay=float(guidance_interval_decay),
            min_guidance_scale=float(min_cfg_eff),

            manual_seeds=manual_seeds,
            retake_seeds=retake_seeds,
            retake_variance=float(retake_variance),

            seam_seconds=float(seam_seconds),
            extend_bootstrap_edge_sec=float(extend_bootstrap_edge_sec),

            lora_name_or_path=lora_name_or_path,
            lora_weight=float(lora_weight),

            batch_size=1,
            debug=False,
        )

        # sanity: ensure the predictor matches the pipeline signature
        accepted = set(inspect.signature(self.pipeline.__call__).parameters.keys())
        unknown = sorted([k for k in kwargs.keys() if k not in accepted])
        if unknown:
            raise RuntimeError(f"predictor kwarg(s) not in pipeline signature: {unknown}")

        output_paths = self.pipeline(**kwargs)
        return Path(output_paths[0])