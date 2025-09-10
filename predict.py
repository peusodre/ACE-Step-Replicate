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
import inspect
from typing import Optional, Dict, Any, List
from cog import BasePredictor, Input, Path

from acestep.pipeline_ace_step import ACEStepPipeline


# ---- Task normalization ------------------------------------------------------

USER_TASK_CHOICES = [
    "text2music",
    "audio2audio",
    "extend",     # (aka continuation)
    "repaint",    # (aka inpainting)
    "style_transfer",
]

def canonicalize_task(user_task: str) -> str:
    if user_task == "extend":
        return "extend"
    if user_task == "repaint":
        return "repaint"
    if user_task == "style_transfer":
        return "text2music"   # <-- was "edit"
    if user_task == "audio2audio":
        return "text2music"    # uses ref_audio_input under the hood
    return "text2music"


# ---- Signature-aware remapping ----------------------------------------------

# For different ACE-Step forks, __call__ may use alternate kwarg names.
KWARG_ALIASES: Dict[str, List[str]] = {
    # logical key -> list of possible arg names in priority order
    "task": ["task", "mode", "task_name", "task_type"],
    "src_audio_path": ["src_audio_path", "src_audio", "src_path", "source_audio_path"],
    "repaint_start": ["repaint_start", "start", "edit_start", "inpaint_start"],
    "repaint_end": ["repaint_end", "end", "edit_end", "inpaint_end"],
    "audio_duration": ["audio_duration", "duration", "total_length"],
    "prompt": ["prompt", "tags", "text_prompt"],
    "lyrics": ["lyrics", "lyric", "text_lyrics"],
    "infer_step": ["infer_step", "infer_steps", "num_inference_steps"],
    "guidance_scale": ["guidance_scale", "cfg_scale"],
    "scheduler_type": ["scheduler_type", "scheduler"],
    "cfg_type": ["cfg_type", "guidance_type"],
    "omega_scale": ["omega_scale", "omega"],
    "manual_seeds": ["manual_seeds", "seed_list", "seeds"],
    "guidance_interval": ["guidance_interval"],
    "guidance_interval_decay": ["guidance_interval_decay"],
    "min_guidance_scale": ["min_guidance_scale", "min_cfg_scale"],
    "use_erg_tag": ["use_erg_tag"],
    "use_erg_lyric": ["use_erg_lyric"],
    "use_erg_diffusion": ["use_erg_diffusion"],
    "guidance_scale_text": ["guidance_scale_text"],
    "guidance_scale_lyric": ["guidance_scale_lyric"],
    "audio2audio_enable": ["audio2audio_enable", "a2a_enable"],
    "ref_audio_input": ["ref_audio_input", "reference_audio", "ref_audio_path"],
    "ref_audio_strength": ["ref_audio_strength", "reference_strength"],
    "retake_seeds": ["retake_seeds", "variation_seeds"],
    "retake_variance": ["retake_variance", "variation_strength"],
    "edit_target_prompt": ["edit_target_prompt", "target_prompt"],
    "edit_target_lyrics": ["edit_target_lyrics", "target_lyrics"],
    "edit_n_min": ["edit_n_min", "n_min"],
    "edit_n_max": ["edit_n_max", "n_max"],
    "edit_n_avg": ["edit_n_avg", "n_avg"],
    "lora_name_or_path": ["lora_name_or_path", "lora", "lora_path"],
    "lora_weight": ["lora_weight"],
    "save_path": ["save_path", "output_dir"],
    "batch_size": ["batch_size"],
    "debug": ["debug"],
    "oss_steps": ["oss_steps"],
}

def remap_to_signature(params: Dict[str, Any], callables_params: List[str]) -> Dict[str, Any]:
    """
    Given our canonical `params` and the actual __call__ param names,
    produce a dict with the best-matching names per alias list.
    """
    accepted = set(callables_params)
    remapped: Dict[str, Any] = {}
    used_targets = set()

    # First pass: try to map all known logical keys via alias list
    for logical_key, value in params.items():
        if logical_key in KWARG_ALIASES:
            for alt in KWARG_ALIASES[logical_key]:
                if alt in accepted and alt not in used_targets:
                    remapped[alt] = value
                    used_targets.add(alt)
                    break
        else:
            # If logical_key itself is directly accepted, keep it
            if logical_key in accepted and logical_key not in used_targets:
                remapped[logical_key] = value
                used_targets.add(logical_key)

    # Second pass: include any additional keys that already exactly match signature
    for k, v in params.items():
        if k in accepted and k not in used_targets:
            remapped[k] = v
            used_targets.add(k)

    # Debug print to see what we ended up sending
    print("DEBUG accepted __call__ kwargs:", sorted(accepted))
    print("DEBUG remapped kwargs keys:", sorted(remapped.keys()))
    # Helpful: show mapped 'task' and 'src_audio_path' equivalents
    task_key = next((alt for alt in KWARG_ALIASES["task"] if alt in remapped), None)
    src_key = next((alt for alt in KWARG_ALIASES["src_audio_path"] if alt in remapped), None)
    print("DEBUG task key/value:", task_key, remapped.get(task_key) if task_key else None)
    print("DEBUG src key/value:", src_key, remapped.get(src_key) if src_key else None)

    return remapped


class Predictor(BasePredictor):
    def setup(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Optimize for faster loading
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

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
    guidance_interval: float = Input(default=0.5, ge=0.0, le=1.0),
    guidance_interval_decay: float = Input(default=0.0, ge=0.0, le=5.0),
    min_guidance_scale: float = Input(default=3.0, ge=0.0, le=30.0),
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

        # Validate
        self._validate_inputs(
            canonical_task,
            input_audio_path,
            reference_audio_path,
            inpaint_start_time,
            inpaint_end_time,
        )

        # Compose prompt/lyrics
        final_prompt = style_prompt if task == "style_transfer" and style_prompt else prompt
        final_lyrics = style_lyrics if task == "style_transfer" and style_lyrics else lyrics

        # Seeds
        manual_seeds = [seed] if seed is not None else None
        retake_seeds = [variation_seed] if variation_seed is not None else manual_seeds
        
        # Duration (for tasks that depend on src audio)
        actual_audio_duration = audio_duration
        if input_audio_path and canonical_task in ("extend", "repaint", "edit"):
            try:
                import librosa
                actual_audio_duration = float(librosa.get_duration(path=input_audio_path))
            except Exception:
                # fallback using torchaudio
                import torchaudio
                info = torchaudio.info(input_audio_path)
                actual_audio_duration = info.num_frames / float(info.sample_rate)

        # Task specifics
        task_kwargs = {
            "task": canonical_task,
            "src_audio_path": None,
            "repaint_start": 0.0,
            "repaint_end": 0.0,
        }

        if canonical_task == "extend":
            # Cap overrun check
            # fps must reflect MusicDCAE native rate when extending a real clip
            fps = float(44100) / (512.0 * 8.0)  # 44.1k native
            def sec_to_frames(sec: float) -> int:
                return int(round(sec * fps))
            want = sec_to_frames(actual_audio_duration + 2.0 * float(extend_duration))
            cap = sec_to_frames(240.0)
            if want > cap:
                raise ValueError(f"Extend exceeds cap: want={want} frames ({actual_audio_duration + 2.0 * float(extend_duration):.1f}s), cap={cap} frames (240.0s). Try smaller extend_duration.")
            
            task_kwargs.update({
                "src_audio_path": input_audio_path,
                "repaint_start": -float(extend_duration),
                "repaint_end": actual_audio_duration + float(extend_duration),
            })
            audio_duration_out = actual_audio_duration + 2.0 * float(extend_duration)
            
            # Extend-specific presets for better quality
            guidance_scale = 20.0
            guidance_interval = 0.6
            min_guidance_scale = 6.0
            cfg_type = "apg"
            use_erg_diffusion = False
            use_erg_lyric = False
            scheduler_type = "heun"                 # more stable transport
            infer_steps = max(int(infer_steps), 90) # enough steps for long pads
            omega_scale = min(float(omega_scale), 8.0)
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
            audio_duration_out = audio_duration

        # Canonical pipeline kwargs (we will remap to actual signature)
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
            # use the possibly-updated values
            "guidance_interval": float(guidance_interval),
            "guidance_interval_decay": float(guidance_interval_decay),
            "min_guidance_scale": float(min_guidance_scale),
            "use_erg_tag": bool(use_erg_tag),
            "use_erg_lyric": bool(use_erg_lyric),
            "use_erg_diffusion": bool(use_erg_diffusion),
            "guidance_scale_text": float(guidance_scale_text),
            "guidance_scale_lyric": float(guidance_scale_lyric),

            "task": task_kwargs["task"],
            "src_audio_path": task_kwargs.get("src_audio_path"),
            "repaint_start": task_kwargs.get("repaint_start", 0.0),
            "repaint_end": task_kwargs.get("repaint_end", 0.0),

            "audio2audio_enable": (task in ("audio2audio", "style_transfer")) and (canonical_task == "text2music"),
            "ref_audio_input": (
                input_audio_path if task == "audio2audio"
                else (reference_audio_path if task == "style_transfer" else None)
            ),
            "ref_audio_strength": (
                float(1.0 - audio2audio_strength) if task == "audio2audio"
                else float(1.0 - style_strength) if task == "style_transfer"
                else 0.5
            ),

            "retake_seeds": retake_seeds,
            "retake_variance": (
                float(repaint_strength) if canonical_task == "repaint"
                else (float(variation_strength) if variation_strength and variation_strength > 0
                      else (max(0.25, min(0.35, 0.3 + 0.1 * float(extend_strength))) if canonical_task == "extend"
                            else float(variation_strength)))
            ),

            "lora_name_or_path": lora_name_or_path,
            "lora_weight": float(lora_weight),
            "save_path": temp_dir,
            "batch_size": 1,
            "debug": False,
            "oss_steps": [],
            "extend_strength": float(extend_strength),  # pass it through
            "sample_rate": int(sample_rate),  # pass sample rate through
        }

        if pipeline_params["src_audio_path"]:
            print("DEBUG exists(src_audio_path):", os.path.exists(pipeline_params["src_audio_path"]))

        # ---- Remap & prune to match the actual __call__ signature ------------
        try:
            sig = inspect.signature(self.pipeline.__call__)
            accepted_names = list(sig.parameters.keys())
            filtered_params = remap_to_signature(pipeline_params, accepted_names)

            # Extra sanity: if 'task' (or alias) isn’t accepted, warn loudly.
            if not any(name in filtered_params for name in KWARG_ALIASES["task"]):
                print("WARNING: No accepted 'task' kwarg name found in signature. Default inside pipeline may be 'text2music'.")

            output_paths = self.pipeline(**filtered_params)
            return Path(output_paths[0])
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