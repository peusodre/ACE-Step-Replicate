"""
Replicate prediction interface for ACE-Step music generation model.
De-duplicated tasks:
- 'extend' (covers 'continuation')
- 'repaint' (covers 'inpainting')
Also supports: text2music, audio2audio, style_transfer (maps to 'edit')
"""

import os
import tempfile
import torch
from typing import Optional
from cog import BasePredictor, Input, Path

from acestep.pipeline_ace_step import ACEStepPipeline


# ---- Task normalization ------------------------------------------------------

# Only expose a minimal, unambiguous set
USER_TASK_CHOICES = [
    "text2music",
    "audio2audio",
    "extend",     # (aka continuation)
    "repaint",    # (aka inpainting)
    "style_transfer",
]

def canonicalize_task(user_task: str) -> str:
    """
    Map user-facing task to the canonical pipeline task names.
    """
    if user_task == "extend":
        return "extend"
    if user_task == "repaint":
        return "repaint"
    if user_task == "style_transfer":
        return "edit"          # edit an existing clip
    if user_task == "audio2audio":
        return "text2music"    # uses ref_audio_input under the hood
    return "text2music"


class Predictor(BasePredictor):
    def setup(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Conservative dtype selection
        dtype = "bfloat16"
        try:
            if torch.backends.mps.is_available():
                dtype = "float32"
            elif torch.cuda.is_available() and not torch.is_bf16_supported():
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

    def predict(
        self,
        # ---- Core UX inputs (de-duplicated) ----------------------------------
        task: str = Input(
            description="Generation task type",
            default="text2music",
            choices=USER_TASK_CHOICES,
        ),
        prompt: str = Input(
            description="Text prompt describing the music",
            default="Electronic dance music, upbeat, energetic",
        ),
        lyrics: str = Input(
            description="Lyrics (optional). Use tags like [verse], [chorus], [bridge]",
            default="",
        ),
        audio_duration: float = Input(
            description="Target output duration (sec)",
            default=30.0, ge=10.0, le=240.0,
        ),

        # ---- Audio inputs ----------------------------------------------------
        input_audio: Path = Input(
            description="Source audio for extend/repaint/edit or as ref for audio2audio",
            default=None,
        ),
        reference_audio: Path = Input(
            description="Additional reference audio (optional; used for style transfer)",
            default=None,
        ),

        # ---- Repaint / Extend specifics --------------------------------------
        inpaint_start_time: float = Input(
            description="Start (sec) of repaint range [repaint only]",
            default=0.0, ge=0.0,
        ),
        inpaint_end_time: float = Input(
            description="End (sec) of repaint range [repaint only]",
            default=0.0, ge=0.0,
        ),
        repaint_strength: float = Input(
            description="How strongly to change the repainted region (0..1)",
            default=0.5, ge=0.0, le=1.0,
        ),
        extend_duration: float = Input(
            description="Seconds to extend on both sides [extend only]",
            default=30.0, ge=5.0, le=120.0,
        ),
        extend_strength: float = Input(
            description="How closely to follow source style in extend (0..1)",
            default=0.7, ge=0.0, le=1.0,
        ),

        # ---- Style transfer (edit) -------------------------------------------
        style_prompt: str = Input(
            description="Style target prompt (optional override for edit)",
            default="",
        ),
        style_lyrics: str = Input(
            description="Style target lyrics (optional override for edit)",
            default="",
        ),
        style_strength: float = Input(
            description="Style transfer strength (0..1). Higher = more change.",
            default=0.7, ge=0.0, le=1.0,
        ),

        # ---- Audio2Audio -----------------------------------------------------
        audio2audio_strength: float = Input(
            description="Audio2Audio generation strength (0..1). Higher = more change.",
            default=0.5, ge=0.0, le=1.0,
        ),

        # ---- Diffusion / guidance --------------------------------------------
        infer_steps: int = Input(default=60, ge=20, le=100),
        guidance_scale: float = Input(default=15.0, ge=1.0, le=30.0),
        scheduler_type: str = Input(default="euler", choices=["euler", "heun", "pingpong"]),
        cfg_type: str = Input(default="apg", choices=["apg", "cfg", "cfg_star"]),
        omega_scale: float = Input(default=10.0, ge=1.0, le=20.0),
        use_erg_tag: bool = Input(default=True),
        use_erg_lyric: bool = Input(default=True),
        use_erg_diffusion: bool = Input(default=True),
        guidance_scale_text: float = Input(default=0.0, ge=0.0, le=30.0),
        guidance_scale_lyric: float = Input(default=0.0, ge=0.0, le=30.0),

        # ---- Reproducibility -------------------------------------------------
        seed: int = Input(default=None),
        variation_seed: int = Input(default=None),
        variation_strength: float = Input(default=0.0, ge=0.0, le=1.0),

        # ---- LoRA / output ---------------------------------------------------
        lora_name_or_path: str = Input(default="none"),
        lora_weight: float = Input(default=1.0, ge=0.0, le=2.0),
        output_format: str = Input(default="wav", choices=["wav", "mp3", "ogg"]),
        sample_rate: int = Input(default=48000, choices=[44100, 48000]),
    ) -> Path:
        temp_dir = tempfile.mkdtemp()

        # Normalize task once; this is the *only* place we decide the canonical task.
        canonical_task = canonicalize_task(task)

        # Materialize audio files (as plain string paths)
        input_audio_path = None
        reference_audio_path = None
        if input_audio is not None:
            input_audio_path = os.path.join(temp_dir, "input_audio.wav")
            with open(input_audio_path, "wb") as f, open(input_audio, "rb") as fi:
                f.write(fi.read())
        if reference_audio is not None:
            reference_audio_path = os.path.join(temp_dir, "reference_audio.wav")
            with open(reference_audio_path, "wb") as f, open(reference_audio, "rb") as fr:
                f.write(fr.read())

        # Validate user intent against required inputs
        self._validate_inputs(
            canonical_task,
            input_audio_path,
            reference_audio_path,
            inpaint_start_time,
            inpaint_end_time,
        )

        # Compose prompt/lyrics (style transfer uses overrides)
        final_prompt = style_prompt if canonical_task == "edit" and style_prompt else prompt
        final_lyrics = style_lyrics if canonical_task == "edit" and style_lyrics else lyrics

        # Seeds
        manual_seeds = [seed] if seed is not None else None
        retake_seeds = [variation_seed] if variation_seed is not None else manual_seeds

        # Compute actual duration for tasks that depend on src audio
        actual_audio_duration = audio_duration
        if input_audio_path and canonical_task in ("extend", "repaint", "edit"):
            import librosa
            actual_audio_duration = float(librosa.get_duration(path=input_audio_path))

        # Build task-specific args (no redundancy)
        task_kwargs = {
            "task": canonical_task,          # canonical value the pipeline expects
            "src_audio_path": None,          # set below when needed
            "repaint_start": 0.0,
            "repaint_end": 0.0,
        }

        if canonical_task == "extend":
            # Extend both sides; bias with ref_audio_input
            task_kwargs.update({
                "src_audio_path": input_audio_path,
                "repaint_start": -float(extend_duration),
                "repaint_end": actual_audio_duration + float(extend_duration),
            })
            # total output duration is symmetric extend
            audio_duration_out = actual_audio_duration + 2.0 * float(extend_duration)
        elif canonical_task == "repaint":
            task_kwargs.update({
                "src_audio_path": input_audio_path,
                "repaint_start": float(inpaint_start_time),
                "repaint_end": float(inpaint_end_time),
            })
            audio_duration_out = actual_audio_duration
        elif canonical_task == "edit":
            task_kwargs.update({
                "src_audio_path": input_audio_path,
            })
            audio_duration_out = actual_audio_duration
        else:
            # text2music / audio2audio
            audio_duration_out = audio_duration

        # Minimal, consistent pipeline kwargs
        pipeline_params = {
            "format": output_format,
            "audio_duration": audio_duration_out,
            "prompt": final_prompt,
            "lyrics": final_lyrics,
            "infer_step": int(infer_steps),
            "guidance_scale": float(guidance_scale),
            "scheduler_type": scheduler_type,
            "cfg_type": cfg_type,
            "omega_scale": float(omega_scale),
            "manual_seeds": manual_seeds,
            "guidance_interval": 0.5,
            "guidance_interval_decay": 0.0,
            "min_guidance_scale": 3.0,
            "use_erg_tag": bool(use_erg_tag),
            "use_erg_lyric": bool(use_erg_lyric),
            "use_erg_diffusion": bool(use_erg_diffusion),
            "guidance_scale_text": float(guidance_scale_text),
            "guidance_scale_lyric": float(guidance_scale_lyric),

            # Canonical task & (compat alias) task_type — ONLY intentional duplication
            "task": task_kwargs["task"],
            "task_type": task_kwargs["task"],

            # Source region args (no-op if not used by the task)
            "src_audio_path": task_kwargs.get("src_audio_path"),
            "repaint_start": task_kwargs.get("repaint_start", 0.0),
            "repaint_end": task_kwargs.get("repaint_end", 0.0),

            # Audio2Audio/style-conditioning knobs (set only when meaningful)
            "audio2audio_enable": (task in ("audio2audio", "extend")),
            "ref_audio_input": input_audio_path if task in ("audio2audio", "extend", "style_transfer") else None,
            "ref_audio_strength": (
                float(1.0 - audio2audio_strength) if task == "audio2audio"
                else float(extend_strength) if task == "extend"
                else 0.3 if task == "style_transfer" and reference_audio_path
                else 0.5
            ),

            # Variation / repaint variance
            "retake_seeds": retake_seeds,
            "retake_variance": (
                float(repaint_strength) if canonical_task == "repaint"
                else float(variation_strength)
            ),

            # LoRA / output / misc
            "lora_name_or_path": lora_name_or_path,
            "lora_weight": float(lora_weight),
            "save_path": temp_dir,
            "batch_size": 1,
            "debug": False,
            "oss_steps": [],
        }

        # Final existence sanity for src path (useful when debugging envs)
        if pipeline_params["src_audio_path"]:
            print("DEBUG exists(src_audio_path):", os.path.exists(pipeline_params["src_audio_path"]))

        try:
            output_paths = self.pipeline(**pipeline_params)
            return Path(output_paths[0])  # [audio_path, params_json]
        except Exception as e:
            raise RuntimeError(f"Prediction failed for task '{task}': {e}")

    # ---- Validation ----------------------------------------------------------

    def _validate_inputs(
        self,
        canonical_task: str,
        input_audio_path: Optional[str],
        reference_audio_path: Optional[str],
        inpaint_start_time: float,
        inpaint_end_time: float,
    ) -> None:
        if canonical_task in ("extend", "repaint", "edit") and not input_audio_path:
            raise ValueError(f"Task '{canonical_task}' requires input_audio")
        if canonical_task == "repaint" and inpaint_end_time <= inpaint_start_time:
            raise ValueError("inpaint_end_time must be greater than inpaint_start_time")