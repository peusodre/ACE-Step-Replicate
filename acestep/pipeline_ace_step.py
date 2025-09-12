"""
ACE-Step pipeline (focused: extend + repaint only)

- 'repaint'   : inpaint/retake a time window inside an existing clip
- 'extend'    : right-only extension to a target total duration, with seam-aware repaint

Dependencies (same as original project):
- acestep.models.ace_step_transformer.ACEStepTransformer2DModel
- acestep.music_dcae.music_dcae_pipeline.MusicDCAE
- acestep.schedulers.* (FlowMatchEuler/Heun/PingPong)
- transformers (UMT5EncoderModel, AutoTokenizer)
- torchaudio, torch
- .cpu_offload (decorator; optional but kept)

Apache 2.0
"""

import os
import re
import math
import time
import json
from dataclasses import dataclass
from typing import Optional, List

import torch
import torchaudio
from loguru import logger
from tqdm import tqdm
from huggingface_hub import snapshot_download

from transformers import UMT5EncoderModel, AutoTokenizer
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
from acestep.models.ace_step_transformer import ACEStepTransformer2DModel
from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from acestep.schedulers.scheduling_flow_match_heun_discrete import FlowMatchHeunDiscreteScheduler
from acestep.schedulers.scheduling_flow_match_pingpong import FlowMatchPingPongScheduler
from acestep.apg_guidance import apg_forward, MomentumBuffer, cfg_forward, cfg_zero_star
from acestep.language_segmentation import LangSegment, language_filters
from acestep.models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer

# NOTE: Use a non-relative import so this file can run both
# as a package module *and* as a flat script in Replicate/Cog.
# If your project is packaged and this file is inside the
# 'acestep' package, you may also use:
#   from acestep.cpu_offload import cpu_offload
# but avoid a leading dot unless you are sure package imports
# are set up correctly.
try:
    from cpu_offload import cpu_offload
except ImportError:
    from .cpu_offload import cpu_offload


# ---- torch runtime knobs -----------------------------------------------------

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---- small utils -------------------------------------------------------------

def ensure_directory_exists(directory: str):
    directory = str(directory)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def _frames_per_second(sample_rate: int) -> float:
    """MusicDCAE latent fps: (SR / (HOP * RATIO)) with HOP=512, RATIO=8."""
    return float(sample_rate) / (512.0 * 8.0)

def _rms(v: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.sqrt(torch.clamp((v.float() ** 2).mean(), eps)).detach().cpu())

def _db(x: float, floor: float = -80.0) -> float:
    if x <= 0:
        return floor
    return max(floor, 20.0 * math.log10(x))

def _avg_time(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Tiny smoothing across time axis for bootstrap tiles."""
    B, C, H, T = x.shape
    y = x.to(torch.float32).view(B * C, H, T)
    y = torch.nn.functional.avg_pool1d(y, kernel_size=k, stride=1, padding=k // 2)
    return y.view(B, C, H, T).to(x.dtype)

def _match_rms_like(z: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    def _r(x): return torch.sqrt(torch.clamp((x * x).mean(dim=-1, keepdim=True), eps))
    return z * (_r(ref) / (_r(z) + eps))

def _match_global_stats(z: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m_ref = ref.mean(dim=(1, 2, 3), keepdim=True)
    s_ref = ref.std(dim=(1, 2, 3), keepdim=True) + eps
    m_z = z.mean(dim=(1, 2, 3), keepdim=True)
    s_z = z.std(dim=(1, 2, 3), keepdim=True) + eps
    return (z - m_z) * (s_ref / s_z) + m_ref


# ---- language / lyrics -------------------------------------------------------

SUPPORT_LANGUAGES = {
    "en": 259, "de": 260, "fr": 262, "es": 284, "it": 285, "pt": 286, "pl": 294,
    "tr": 295, "ru": 267, "cs": 293, "nl": 297, "ar": 5022, "zh": 5023, "ja": 5412,
    "hu": 5753, "ko": 6152, "hi": 6680,
}
_structure_re = re.compile(r"\[.*?\]")

REPO_ID = "ACE-Step/ACE-Step-v1-3.5B"
REPO_ID_QUANT = REPO_ID + "-q4-K-M"  # not used here, but kept for parity


@dataclass
class DebugCfg:
    enabled: bool = False
    step_every: int = 10
    run_tag: str = ""


# ---- pipeline ----------------------------------------------------------------

class ACEStepPipeline:
    """
    Focused pipeline exposing only two high-level tasks:
      - repaint: modify/inpaint a time window [repaint_start, repaint_end] in the source
      - extend : right-only extension to a target total duration by tiling an edge motif,
                 then running a masked repaint over the padded tail
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        device_id: int = 0,
        dtype: str = "bfloat16",
        persistent_storage_path: Optional[str] = None,
        torch_compile: bool = False,
        cpu_offload: bool = False,
        quantized: bool = False,
        overlapped_decode: bool = False,
        **_: dict,
    ):
        # where weights live
        if not checkpoint_dir:
            if persistent_storage_path is None:
                checkpoint_dir = os.path.join(os.path.expanduser("~"), ".cache/ace-step/checkpoints")
            else:
                checkpoint_dir = os.path.join(persistent_storage_path, "checkpoints")
        ensure_directory_exists(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir

        # device/dtype
        device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cpu" and torch.backends.mps.is_available():
            device = torch.device("mps")
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        if device.type == "mps":
            self.dtype = torch.float32
        self.device = device

        self.loaded = False
        self.torch_compile = torch_compile
        self.cpu_offload = cpu_offload
        self.quantized = quantized
        self.overlapped_decode = overlapped_decode

        self.lora_path = "none"
        self.lora_weight = 1.0

        # text/lyrics helpers
        self.lang_segment = LangSegment()
        self.lang_segment.setfilters(language_filters.default)
        self.lyric_tokenizer = VoiceBpeTokenizer()

        self.debug = DebugCfg(False)

    # ---- checkpoint load -----------------------------------------------------

    def _resolve_repo_cache(self, checkpoint_dir: Optional[str], repo_id: str) -> str:
        """Resolve local checkpoint cache (or download into cache)."""
        checkpoint_dir_models = None
        if checkpoint_dir is not None:
            required = ["music_dcae_f8c8", "music_vocoder", "ace_step_transformer", "umt5-base"]
            if all(os.path.exists(os.path.join(checkpoint_dir, d)) for d in required):
                logger.info(f"Using local checkpoints: {checkpoint_dir}")
                checkpoint_dir_models = checkpoint_dir
        
        if checkpoint_dir_models is None:
            if checkpoint_dir is None:
                logger.info(f"Downloading models from HF: {repo_id}")
                checkpoint_dir_models = snapshot_download(repo_id)
            else:
                logger.info(f"Downloading models from HF: {repo_id} → {checkpoint_dir}")
                checkpoint_dir_models = snapshot_download(repo_id, cache_dir=checkpoint_dir)

        return checkpoint_dir_models

    def load_checkpoint(self, checkpoint_dir: Optional[str] = None):
        ckdir = self._resolve_repo_cache(checkpoint_dir or self.checkpoint_dir, REPO_ID)
        dcae_path = os.path.join(ckdir, "music_dcae_f8c8")
        voc_path = os.path.join(ckdir, "music_vocoder")
        ace_path = os.path.join(ckdir, "ace_step_transformer")
        txt_path = os.path.join(ckdir, "umt5-base")

        # core models
        self.ace_step_transformer = ACEStepTransformer2DModel.from_pretrained(ace_path, torch_dtype=self.dtype)
        self.music_dcae = MusicDCAE(dcae_checkpoint_path=dcae_path, vocoder_checkpoint_path=voc_path)
        self.text_encoder_model = UMT5EncoderModel.from_pretrained(txt_path, torch_dtype=self.dtype).eval()
        self.text_tokenizer = AutoTokenizer.from_pretrained(txt_path)

        # place on device / dtype
        tgt = "cpu" if self.cpu_offload else self.device
        self.ace_step_transformer = self.ace_step_transformer.to(tgt).eval().to(self.dtype)
        self.music_dcae = self.music_dcae.to(tgt).eval().to(self.dtype)
        self.text_encoder_model = self.text_encoder_model.to(tgt).eval().to(self.dtype)
        self.text_encoder_model.requires_grad_(False)

        # optional compile
        if self.torch_compile and self.device.type == "cuda" and not self.quantized:
            self.ace_step_transformer = torch.compile(self.ace_step_transformer)
            self.music_dcae = torch.compile(self.music_dcae)
            self.text_encoder_model = torch.compile(self.text_encoder_model)

        self.loaded = True
        logger.info(f"[ENV] device={self.device} dtype={self.dtype} torch={torch.__version__}")

    # ---- memory hygiene ------------------------------------------------------

    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        import gc
        gc.collect()

    # ---- text / lyrics encoders ---------------------------------------------

    @cpu_offload("text_encoder_model")
    def get_text_embeddings(self, texts: List[str], text_max_length: int = 256):
        tok = self.text_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=text_max_length
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}
        if self.text_encoder_model.device != self.device:
            self.text_encoder_model.to(self.device)
        with torch.no_grad():
            out = self.text_encoder_model(**tok)
        return out.last_hidden_state, tok["attention_mask"]

    def get_lang(self, text: str) -> str:
        lang = "en"
        try:
            _ = self.lang_segment.getTexts(text)
            counts = self.lang_segment.getCounts()
            lang = counts[0][0]
            if len(counts) > 1 and lang == "en":
                lang = counts[1][0]
        except Exception:
            lang = "en"
        return "zh" if "zh" in lang else ("es" if "spa" in lang else (lang if lang in SUPPORT_LANGUAGES else "en"))

    def tokenize_lyrics(self, lyrics: str, debug: bool = False) -> List[int]:
        if not lyrics:
            return [0]
        lines = lyrics.split("\n")
        ids = [261]
        for line in lines:
            s = line.strip()
            if not s:
                ids += [2]
                continue
            lang = self.get_lang(s)
            try:
                tks = self.lyric_tokenizer.encode(s, "en" if _structure_re.match(s) else lang)
                if debug:
                    toks = self.lyric_tokenizer.batch_decode([[tid] for tid in tks])
                    logger.info(f"[LYR] {s} -> {lang} -> {toks}")
                ids = ids + tks + [2]
            except Exception as e:
                logger.warning(f"tokenize error {e} for line {s} lang={lang}")
        return ids

    # ---- random seeds --------------------------------------------------------

    def set_seeds(self, batch_size: int, manual_seeds: Optional[List[int]] = None):
        def _normalize(inp):
            if inp is None:
                return None
            if isinstance(inp, int):
                return [inp]
            if isinstance(inp, str):
                return [int(s) for s in inp.split(",")] if "," in inp else [int(inp)]
            if isinstance(inp, (list, tuple)):
                return [int(s) for s in inp]
            return None

        seeds = _normalize(manual_seeds)
        gens = [torch.Generator(device=self.device) for _ in range(batch_size)]
        actual = []
        for i in range(batch_size):
            s = seeds[i] if (seeds and i < len(seeds)) else (seeds[-1] if seeds else torch.randint(0, 2**32, (1,)).item())
            gens[i].manual_seed(int(s))
            actual.append(int(s))
        return gens, actual

    # ---- encode / decode latents --------------------------------------------

    @cpu_offload("music_dcae")
    def infer_latents(self, input_audio_path: Optional[str]) -> Optional[torch.Tensor]:
        if input_audio_path is None:
            return None
        try:
            audio, sr = self.music_dcae.load_audio(input_audio_path)
            audio = audio.unsqueeze(0).to(device=self.device, dtype=self.dtype)
            latents, _ = self.music_dcae.encode(audio, sr=sr)
            return latents
        except Exception as e:
            logger.error(f"Failed to load/encode audio from {input_audio_path}: {e}")
            return None

    @cpu_offload("music_dcae")
    def latents2audio(
        self,
        latents: torch.Tensor,
        target_wav_duration_second: float,
        sample_rate: int = 48000,
        save_path: Optional[str] = None,
        format: str = "wav",
    ) -> List[str]:
        bs = latents.shape[0]
        with torch.no_grad():
            if self.overlapped_decode and target_wav_duration_second > 48:
                _, wavs = self.music_dcae.decode_overlap(latents, sr=sample_rate)
            else:
                _, wavs = self.music_dcae.decode(latents, sr=sample_rate)
        wavs = [w.cpu().float() for w in wavs]

        paths: List[str] = []
        for i in tqdm(range(bs), disable=(bs <= 1)):
            paths.append(self._save_wav(wavs[i], i, save_path, sample_rate, format))
        return paths

    def _save_wav(self, wav: torch.Tensor, idx: int, save_path: Optional[str], sample_rate: int, format: str) -> str:
        if save_path is None:
            base = "./outputs"
            ensure_directory_exists(base)
            out = f"{base}/output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}"
        else:
            ensure_directory_exists(os.path.dirname(save_path) or ".")
            out = (
                os.path.join(save_path, f"output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}")
                if os.path.isdir(save_path)
                else save_path
            )

        fmt = (format or "wav").lower()
        try:
            if fmt in ("wav", "ogg"):
                torchaudio.save(out, wav.float(), sample_rate=sample_rate, format=fmt)
            elif fmt == "mp3":
                try:
                    torchaudio.save(out, wav.float(), sample_rate=sample_rate, format="mp3")
                except Exception as e:
                    logger.warning(f"MP3 save failed ({e}); falling back to WAV.")
                    out = out.rsplit(".", 1)[0] + ".wav"
                    torchaudio.save(out, wav.float(), sample_rate=sample_rate, format="wav")
            else:
                logger.warning(f"Unknown format '{format}', saving as WAV instead.")
                out = out.rsplit(".", 1)[0] + ".wav"
                torchaudio.save(out, wav.float(), sample_rate=sample_rate, format="wav")
        except Exception as e:
            logger.warning(f"Primary save failed ({e}); attempting WAV fallback.")
            try:
                out = out.rsplit(".", 1)[0] + ".wav"
            except Exception:
                out = f"{out}.wav"
            torchaudio.save(out, wav.float(), sample_rate=sample_rate, format="wav")

        logger.info(f"Saved audio to {out}")
        return out

    # ---- core diffusion (extend/repaint) ------------------------------------

    @cpu_offload("ace_step_transformer")
    @torch.no_grad()
    def _extend_repaint_diffusion(
        self,
        *,
        task: str,                      # "extend" | "repaint"
        src_latents: torch.Tensor,      # (B,8,16,Tsrc)
        repaint_start_sec: float,       # repaint only; ignored for extend
        repaint_end_sec: float,         # repaint: end sec; extend: TARGET TOTAL seconds
        prompt_hid: torch.Tensor,       # (B, L, D)
        prompt_mask: torch.Tensor,      # (B, L)
        lyric_ids: torch.Tensor,        # (B, L_lyr)
        lyric_mask: torch.Tensor,       # (B, L_lyr)
        infer_steps: int,
        scheduler_type: str,
        cfg_type: str,
        guidance_scale: float,
        omega_scale: float,
        guidance_interval: float,
        guidance_interval_decay: float,
        min_guidance_scale: float,
        retake_random_generators: List[torch.Generator],
        retake_variance: float,
        seam_seconds: float,
        sample_rate: int,
        extend_bootstrap_edge_sec: float,
    ) -> torch.Tensor:

        # scheduler
        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        elif scheduler_type == "pingpong":
            scheduler = FlowMatchPingPongScheduler(num_train_timesteps=1000, shift=3.0)
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

        # fps for latent timing (MusicDCAE is 44.1k native for src latents)
        fps_src = _frames_per_second(44100)
        fps_out = _frames_per_second(sample_rate) if task == "repaint" else fps_src

        # source & frame length
        x0 = src_latents
        B, C, H, Tsrc = x0.shape
        frame_length = Tsrc

        # map times to frames
        def sec_to_frames(sec: float, fps: float) -> int:
            return int(round(max(0.0, sec) * fps))

        # create initial target latents (pure noise)
        target_latents = randn_tensor(
            shape=(B, 8, 16, frame_length),
            generator=retake_random_generators,
            device=self.device,
            dtype=self.dtype,
        )

        # build repaint/extend mask + seed z0
        repaint_mask = torch.zeros_like(x0)
        z0 = None
        is_extend = False

        if task == "repaint":
            s_f = min(Tsrc, sec_to_frames(repaint_start_sec, fps_out))
            e_f = min(Tsrc, sec_to_frames(repaint_end_sec,   fps_out))
            if e_f <= s_f:
                raise ValueError("repaint_end must be > repaint_start")

            repaint_mask[..., s_f:e_f] = 1.0

            # retake_variance α∈[0,1] controls deviation from source:
            # seed = (1-α)*x0 + α*noise_mix  inside the repaint window
            alpha = float(retake_variance)
            base_seed = x0.clone()  # source latents
            noise_a = target_latents  # current random init
            noise_b = randn_tensor(
                shape=target_latents.shape,
                generator=retake_random_generators,
                device=self.device,
                dtype=self.dtype,
            )
            angle = torch.tensor(alpha * math.pi / 2, device=self.device, dtype=self.dtype)
            noise_mix = torch.cos(angle) * noise_a + torch.sin(angle) * noise_b
            seed_mix = (1.0 - alpha) * base_seed + alpha * noise_mix
            z0 = torch.where(repaint_mask == 1.0, seed_mix, x0)

            # Debug: how much perturbation vs source are we injecting in-window?
            with torch.no_grad():
                m = (repaint_mask > 0).to(self.dtype)
                denom = m.sum().clamp_min(1.0)
                delta = ((z0 - x0) * m).pow(2).sum() / denom
                delta_rms = torch.sqrt(torch.clamp(delta, 1e-12)).item()
                logger.info(f"[REPAINT] noise alpha={alpha:.3f}  window_rms={_db(delta_rms):.1f} dB")
                
                # Sanity check: preserved region should stay put
                keep_m = (repaint_mask == 0).to(self.dtype)
                denom_keep = keep_m.sum().clamp_min(1.0)
                drift = (((z0 - x0) * keep_m).pow(2).sum() / denom_keep).sqrt().item()
                logger.info(f"[REPAINT] preserved_region drift_rms={_db(drift):.1f} dB (should be ≈ -inf)")

            logger.info(f"[REPAINT] frames=({s_f},{e_f}) / {Tsrc} (≈{s_f/fps_out:.2f}s→{e_f/fps_out:.2f}s)")

        else:  # extend (right-only to target total seconds in repaint_end_sec)
            target_total_sec = float(repaint_end_sec)
            want_f = max(Tsrc, sec_to_frames(target_total_sec, fps_src))
            cap_f = int(round(240.0 * fps_src))
            if want_f > cap_f:
                raise ValueError(f"Extend target exceeds cap: want={want_f}f, cap={cap_f}f (240s)")

            # pad tail only
            right_pad = max(0, want_f - Tsrc)
            if right_pad > 0:
                # 1) expand source and prep mask container
                x0 = torch.nn.functional.pad(x0, (0, right_pad), "constant", 0)
                repaint_mask = torch.zeros_like(x0)
                frame_length = x0.shape[-1]
                is_extend = True

                # 2) tile last EDGE seconds as tail motif
                EDGE = max(64, sec_to_frames(extend_bootstrap_edge_sec, fps_src))
                core = x0[..., :Tsrc]
                edge = _avg_time(x0[..., Tsrc-EDGE:Tsrc] if Tsrc >= EDGE else x0[..., :Tsrc], k=7)

                reps = (right_pad + edge.shape[-1] - 1) // edge.shape[-1]
                tiled = edge.repeat(1, 1, 1, reps)[..., :right_pad]

                # 3) latent seed crossfade exactly at seam (helps before denoiser acts)
                ramp_L_seed = min(int(round(seam_seconds * fps_src)), right_pad)
                if ramp_L_seed > 0:
                    fade = torch.linspace(0.0, 1.0, steps=ramp_L_seed,
                                          device=self.device, dtype=self.dtype)[None, None, None, :]
                    cross = (1.0 - fade) * core[..., -ramp_L_seed:] + fade * tiled[..., :ramp_L_seed]
                    tiled = torch.cat([cross, tiled[..., ramp_L_seed:]], dim=-1)

                z0 = torch.cat([core, tiled], dim=-1)

                # keep stats stable for the vocoder
                z0 = _match_rms_like(z0, x0)
                z0 = _match_global_stats(z0, x0)

                target_latents = z0.clone()

                # 4) Seam-aware repaint mask:
                # Make the mask reach 1.0 *at the last source frame* and keep 1.0 for all padded frames.
                # This ensures the denoiser is already at full authority when the tail begins.
                pre_seconds = max(0.0, seam_seconds)
                pre_L = min(int(round(pre_seconds * fps_src)), Tsrc)

                seam_src_start = Tsrc - pre_L
                seam_pad_start = Tsrc

                # Pre-seam ramp **inside the source**: 0 → 1 so the final source frame is exactly 1.0
                if pre_L > 0:
                    pre_ramp = torch.linspace(
                        0.0, 1.0, steps=pre_L, device=self.device, dtype=self.dtype
                    )[None, None, None, :]
                    repaint_mask[..., seam_src_start:Tsrc] = torch.maximum(
                        repaint_mask[..., seam_src_start:Tsrc], pre_ramp
                    )

                # Post-seam (the padded tail): full repaint immediately
                if right_pad > 0:
                    repaint_mask[..., seam_pad_start : seam_pad_start + right_pad] = 1.0

                # Mix for *tail only*, controlled by α=retake_variance, relative to the tiled edge:
                # tail_seed = (1-α)*tiled + α*noise_mix
                alpha = float(retake_variance)
                tail_start = Tsrc
                tail_end   = Tsrc + right_pad
                tail_slice = slice(tail_start, tail_end)

                tiled_seed = target_latents[..., tail_slice]  # current tiled/crossfaded seed
                if alpha > 0.0:
                    noise_b = randn_tensor(
                        shape=tiled_seed.shape,
                        generator=retake_random_generators,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    angle = torch.tensor(alpha * math.pi / 2, device=self.device, dtype=self.dtype)
                    noise_mix = torch.cos(angle) * tiled_seed + torch.sin(angle) * noise_b
                    tail_seed = (1.0 - alpha) * tiled_seed + alpha * noise_mix

                    # Gentle ramp-in of the perturbation right after the seam
                    pre_noise_ramp = min(int(round(0.25 * seam_seconds * fps_src)), right_pad)
                    if pre_noise_ramp > 0:
                        fade = torch.linspace(0.0, 1.0, steps=pre_noise_ramp,
                                              device=self.device, dtype=self.dtype)[None, None, None, :]
                        tail_seed[..., :pre_noise_ramp] = (
                            (1.0 - fade) * tiled_seed[..., :pre_noise_ramp] +
                            fade * tail_seed[..., :pre_noise_ramp]
                        )
                else:
                    tail_seed = tiled_seed

                # Apply the mixed seed to the tail in both target_latents and z0
                target_latents[..., tail_slice] = tail_seed
                z0[...,            tail_slice] = tail_seed

                # Debug: perturbation vs the tiled seed (not vs zeros)
                with torch.no_grad():
                    delta = (tail_seed - tiled_seed).pow(2).mean().sqrt().item()
                    logger.info(f"[EXTEND] noise alpha={alpha:.3f}  tail_rms={_db(delta):.1f} dB")

                logger.info(
                    f"[EXTEND] src={Tsrc}f (≈{Tsrc/fps_src:.2f}s) → new={frame_length}f "
                    f"(+{right_pad}f), seam_pre={pre_L}f (mask hits 1.0 at seam)"
                )
            else:
                repaint_mask = torch.zeros_like(x0)
                z0 = x0.clone()
                target_latents = x0.clone()
                logger.info("[EXTEND] right_pad=0 (target length ≤ source length); no repaint region.")

        # timesteps
        timesteps, num_steps = retrieve_timesteps(
            scheduler, num_inference_steps=infer_steps, device=self.device, timesteps=None
        )

        # build encoder features (single condition stream; ERG off in this minimal variant)
        spk = torch.zeros(B, 512, device=self.device, dtype=self.dtype)
        enc_states, enc_mask = self.ace_step_transformer.encode(
            prompt_hid, prompt_mask, spk, lyric_ids, lyric_mask
        )
        # unconditional stream (zeros)
        enc_states_null, _ = self.ace_step_transformer.encode(
            torch.zeros_like(prompt_hid), prompt_mask, torch.zeros_like(spk),
            torch.zeros_like(lyric_ids), lyric_mask
        )

        # guidance window
        start_idx = int(num_steps * ((1 - guidance_interval) / 2))
        end_idx = int(num_steps * (guidance_interval / 2 + 0.5))
        if is_extend:
            start_idx, end_idx = 0, num_steps
            guidance_interval_decay = 1.2 if guidance_interval_decay == 0.0 else guidance_interval_decay
            min_guidance_scale = max(6.0, float(min_guidance_scale))
            omega_scale = min(10.0, float(omega_scale))

        attn_mask = torch.ones(B, frame_length, device=self.device, dtype=torch.bool)
        momentum = MomentumBuffer()

        # repaint strength → choose step index where masked ODE-like blending begins
        # (no sigma capping/override; this is the only place we decide "when to start")
        repaint_frac = float(retake_variance)
        n_min = max(1, min(int(infer_steps * (1 - repaint_frac)), infer_steps - 2))

        # diffusion
        for i, t in tqdm(enumerate(timesteps), total=num_steps, disable=(num_steps <= 12)):
            latents = target_latents
            in_guidance = (start_idx <= i < end_idx)

            if in_guidance:
                # possibly decaying CFG during the active guidance window
                if guidance_interval_decay > 0:
                    den = max(1, (end_idx - start_idx - 1))
                    prog = (i - start_idx) / den
                    cfg_now = guidance_scale - (guidance_scale - min_guidance_scale) * prog * guidance_interval_decay
                else:
                    cfg_now = guidance_scale

                timestep = t.expand(latents.shape[0])
                out_len = latents.shape[-1]

                # cond / uncond passes
                cond = self.ace_step_transformer.decode(
                    hidden_states=latents, attention_mask=attn_mask,
                    encoder_hidden_states=enc_states, encoder_hidden_mask=enc_mask,
                    output_length=out_len, timestep=timestep
                    ).sample
                uncond = self.ace_step_transformer.decode(
                    hidden_states=latents, attention_mask=attn_mask,
                    encoder_hidden_states=enc_states_null, encoder_hidden_mask=enc_mask,
                    output_length=out_len, timestep=timestep
                    ).sample

                if cfg_type == "apg":
                    noise = apg_forward(
                        pred_cond=cond, pred_uncond=uncond,
                        guidance_scale=cfg_now, momentum_buffer=momentum
                    )
                elif cfg_type == "cfg":
                    noise = cfg_forward(
                        cond_output=cond, uncond_output=uncond, cfg_strength=cfg_now
                    )
                elif cfg_type == "cfg_star":
                    noise = cfg_zero_star(
                        noise_pred_with_cond=cond, noise_pred_uncond=uncond,
                        guidance_scale=cfg_now, i=i, zero_steps=1, use_zero_init=True
                    )
                else:
                    noise = cond  # fallback
            else:
                # unconditional evolution (no explicit guidance mixing)
                timestep = t.expand(latents.shape[0])
                noise = self.ace_step_transformer.decode(
                    hidden_states=latents, attention_mask=attn_mask,
                    encoder_hidden_states=enc_states, encoder_hidden_mask=enc_mask,
                    output_length=latents.shape[-1], timestep=timestep
                ).sample

            # repaint / extend branch: after n_min, do masked ODE-like update and blend
            if (task == "repaint" or is_extend) and i >= n_min:
                t_i = t / 1000
                t_im1 = (timesteps[i + 1] / 1000) if (i + 1 < len(timesteps)) else torch.zeros_like(t_i).to(self.device)

                target_latents = target_latents.to(torch.float32)
                prev = target_latents + (t_im1 - t_i) * noise
                prev = prev.to(self.dtype)
                target_latents = prev

                # "source" latent trajectory at next step
                if z0 is None:
                    z0 = target_latents  # safety
                zt_src = (1 - t_im1) * x0 + (t_im1) * z0
                # soft blend: 1 in repaint tail, 0 in preserved region
                target_latents = repaint_mask * target_latents + (1.0 - repaint_mask) * zt_src
            else:
                # normal diffusion step
                target_latents = scheduler.step(
                    model_output=noise, timestep=t, sample=target_latents,
                    return_dict=False, omega=omega_scale,
                    generator=(retake_random_generators[0] if retake_random_generators else None)
                )[0]

        return target_latents

    # ---- public API ----------------------------------------------------------

    def load_lora(self, lora_name_or_path: str, lora_weight: float):
        if lora_name_or_path == "none":
            if self.lora_path != "none":
                self.ace_step_transformer.unload_lora()
                self.lora_path = "none"
            return

        if lora_name_or_path == self.lora_path and abs(lora_weight - self.lora_weight) < 1e-6:
            return

        # ↓ outdented to function scope (not under the return) ↓
        if not os.path.exists(lora_name_or_path):
            lora_download_path = snapshot_download(lora_name_or_path, cache_dir=self.checkpoint_dir)
        else:
            lora_download_path = lora_name_or_path

        if self.lora_path != "none":
            self.ace_step_transformer.unload_lora()

        self.ace_step_transformer.load_lora_adapter(
            os.path.join(lora_download_path, "pytorch_lora_weights.safetensors"),
            adapter_name="ace_step_lora", with_alpha=True, prefix=None
        )
        from diffusers.utils.peft_utils import set_weights_and_activate_adapters
        set_weights_and_activate_adapters(self.ace_step_transformer, ["ace_step_lora"], [lora_weight])

        self.lora_path = lora_name_or_path
        self.lora_weight = lora_weight
        logger.info(f"[LORA] loaded {lora_name_or_path} (weight={lora_weight})")

    def __call__(
        self,
        *,
        # ---- IO / task -------------------------------------------------------
        format: str = "wav",
        sample_rate: int = 48000,
        save_path: Optional[str] = None,

        task: str = "extend",                       # "extend" | "repaint"
        src_audio_path: Optional[str] = None,       # required for both tasks

        # repaint: [repaint_start, repaint_end] in seconds
        # extend : repaint_end = desired TOTAL seconds (right-only)
        repaint_start: float = 0.0,
        repaint_end: float = 0.0,

        # ---- text conditions -------------------------------------------------
        prompt: str = "",
        lyrics: str = "",

        # ---- diffusion / guidance -------------------------------------------
        infer_step: int = 60,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        guidance_scale: float = 15.0,
        omega_scale: float = 10.0,
        guidance_interval: float = 0.5,
        guidance_interval_decay: float = 1.0,
        min_guidance_scale: float = 3.0,

        # ---- seeds / variation ----------------------------------------------
        manual_seeds: Optional[List[int]] = None,
        retake_seeds: Optional[List[int]] = None,
        retake_variance: float = 0.3,     # repaint strength / extend variation

        # ---- extend controls -------------------------------------------------
        seam_seconds: float = 0.75,
        extend_bootstrap_edge_sec: float = 2.0,

        # ---- LoRA / batching -------------------------------------------------
        lora_name_or_path: str = "none",
        lora_weight: float = 1.0,
        batch_size: int = 1,

        # diagnostics
        debug: bool = False,
    ) -> List[str]:
        """
        Returns: list of output audio paths + trailing JSON dump for reproducibility.
        """
        if task not in ("extend", "repaint"):
            raise ValueError("task must be 'extend' or 'repaint'")
        if not src_audio_path or not os.path.exists(src_audio_path):
            raise ValueError("src_audio_path is required and must exist")

        if not self.loaded:
            logger.info("Loading checkpoints…")
            self.load_checkpoint(self.checkpoint_dir)

        # LoRA
        self.load_lora(lora_name_or_path, float(lora_weight))

        # seeds
        _, actual_seeds = self.set_seeds(batch_size, manual_seeds)
        retake_gen, actual_retake = self.set_seeds(batch_size, retake_seeds if retake_seeds else manual_seeds)

        # text/lyrics enc
        prompt_hid, prompt_mask = self.get_text_embeddings([prompt or ""])
        prompt_hid = prompt_hid.repeat(batch_size, 1, 1)
        prompt_mask = prompt_mask.repeat(batch_size, 1)

        if lyrics:
            lyr = self.tokenize_lyrics(lyrics, debug=debug)
            lyric_ids = torch.tensor(lyr, device=self.device).unsqueeze(0).repeat(batch_size, 1).long()
            lyric_mask = torch.tensor([1] * len(lyr), device=self.device).unsqueeze(0).repeat(batch_size, 1).long()
        else:
            lyric_ids = torch.tensor([0], device=self.device).repeat(batch_size, 1).long()
            lyric_mask = torch.tensor([0], device=self.device).repeat(batch_size, 1).long()

        # src latents
        src_latents = self.infer_latents(src_audio_path)
        if src_latents is None:
            raise ValueError(f"Failed to load audio from {src_audio_path}")

        # run diffusion
        t0 = time.time()
        latents = self._extend_repaint_diffusion(
            task=task,
                src_latents=src_latents,
            repaint_start_sec=float(repaint_start),
            repaint_end_sec=float(repaint_end),
            prompt_hid=prompt_hid,
            prompt_mask=prompt_mask,
            lyric_ids=lyric_ids,
                lyric_mask=lyric_mask,
            infer_steps=int(infer_step),
                scheduler_type=scheduler_type,
                cfg_type=cfg_type,
            guidance_scale=float(guidance_scale),
            omega_scale=float(omega_scale),
            guidance_interval=float(guidance_interval),
            guidance_interval_decay=float(guidance_interval_decay),
            min_guidance_scale=float(min_guidance_scale),
            retake_random_generators=retake_gen,
            retake_variance=float(retake_variance),
            seam_seconds=float(seam_seconds),
            sample_rate=int(sample_rate),
            extend_bootstrap_edge_sec=float(extend_bootstrap_edge_sec),
        )
        logger.info(f"[DIFFUSION] {(time.time() - t0):.2f}s  latents={tuple(latents.shape)}  rms={_db(_rms(latents)):.1f} dB")

        # decode
        # pick output duration: repaint keeps original; extend uses repaint_end (total seconds)
        if task == "repaint":
            # measure source seconds at 44.1k latent fps
            fps_src = _frames_per_second(44100)
            out_sec = src_latents.shape[-1] / fps_src
        else:
            out_sec = float(repaint_end)

        t1 = time.time()
        paths = self.latents2audio(
            latents=latents,
            target_wav_duration_second=out_sec,
            save_path=save_path,
            sample_rate=int(sample_rate),
            format=format,
        )
        logger.info(f"[DECODE] {(time.time() - t1):.2f}s")

        # memory hygiene
        self.cleanup_memory()

        # metadata (one json per file, co-located)
        meta = {
            "task": task,
            "prompt": prompt,
            "lyrics": lyrics,
            "src_audio_path": src_audio_path,
            "repaint_start": float(repaint_start),
            "repaint_end": float(repaint_end),
            "infer_step": int(infer_step),
            "scheduler_type": scheduler_type,
            "cfg_type": cfg_type,
            "guidance_scale": float(guidance_scale),
            "omega_scale": float(omega_scale),
            "guidance_interval": float(guidance_interval),
            "guidance_interval_decay": float(guidance_interval_decay),
            "min_guidance_scale": float(min_guidance_scale),
            "retake_variance": float(retake_variance),
            "seam_seconds": float(seam_seconds),
            "extend_bootstrap_edge_sec": float(extend_bootstrap_edge_sec),
            "lora_name_or_path": lora_name_or_path,
            "lora_weight": float(lora_weight),
            "timecosts": {},
            "actual_seeds": actual_seeds,
            "retake_seeds": actual_retake,
            "audio_path": None,
            "format": format,
            "sample_rate": int(sample_rate),
        }
        for p in paths:
            meta_path = p.replace(f".{format}", "_input_params.json")
            meta["audio_path"] = p
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)

        return paths + [meta]