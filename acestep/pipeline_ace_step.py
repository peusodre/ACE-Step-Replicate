"""
ACE-Step: A Step Towards Music Generation Foundation Model

https://github.com/ace-step/ACE-Step

Apache 2.0 License
"""

import random
import time
import os
import re
import math

import torch
from loguru import logger
from tqdm import tqdm
import json
from huggingface_hub import snapshot_download

# from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from acestep.schedulers.scheduling_flow_match_heun_discrete import (
    FlowMatchHeunDiscreteScheduler,
)
from acestep.schedulers.scheduling_flow_match_pingpong import (
    FlowMatchPingPongScheduler,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from transformers import UMT5EncoderModel, AutoTokenizer

from acestep.language_segmentation import LangSegment, language_filters
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
from acestep.models.ace_step_transformer import ACEStepTransformer2DModel
from acestep.models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
from acestep.apg_guidance import (
    apg_forward,
    MomentumBuffer,
    cfg_forward,
    cfg_zero_star,
    cfg_double_condition_forward,
)
import torchaudio
from .cpu_offload import cpu_offload
from dataclasses import dataclass
from typing import Optional


@dataclass
class DebugCfg:
    enabled: bool = False          # master switch
    step_every: int = 10           # how often to log during diffusion
    dump_specs: bool = False       # save mel/spec PNGs after decode
    dump_latents_npz: bool = False # save .npz of key latents
    dump_audio_wav: bool = False   # save extra debug wavs (pre/post)
    run_tag: str = ""              # optional tag to identify logs

def _rms(x, eps=1e-12):
    return torch.sqrt(torch.clamp((x.float()**2).mean(), eps))

def _rms_t(x, eps=1e-12):
    # RMS over time axis only (last dim), then mean over batch/channels
    return float(torch.sqrt(torch.clamp((x.float()**2).mean(dim=-1), eps)).mean().detach().cpu())

def _db(v, floor=-80.0):
    v = float(v)
    if v <= 0: return floor
    return max(floor, 20.0 * math.log10(v))

def _tstats(name, x):
    if x is None: 
        logger.info(f"[DBG] {name}: None")
        return
    x32 = x.detach().float()
    mn = float(x32.min().cpu())
    mx = float(x32.max().cpu())
    me = float(x32.mean().cpu())
    sd = float(x32.std().cpu())
    rm = float(_rms(x32))
    logger.info(f"[DBG] {name}: shape={tuple(x.shape)} dtype={x.dtype} "
                f"min={mn:.5f} max={mx:.5f} mean={me:.5f} std={sd:.5f} rms={rm:.5f} ({_db(rm):.2f} dB)")

def _center_diff(a, b):
    if a is None or b is None: return None
    d = (a - b).abs().amax().item()
    return d

def _mel_spectrogram_png(wav, sr, outfile):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torchaudio
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128
        )(wav.cpu())
        mel_db = 10.0 * torch.log10(mel.clamp_min(1e-8))
        plt.figure(figsize=(10, 3))
        plt.imshow(mel_db.squeeze(0).numpy(), aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(os.path.basename(outfile))
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()
        logger.info(f"[DBG] wrote {outfile}")
    except Exception as e:
        logger.warning(f"[DBG] mel save failed: {e}")

class _Tick:
    def __init__(self, name): 
        self.name = name
        self.t = time.time()
    def done(self, note=""): 
        dt = time.time() - self.t
        logger.info(f"[TIME] {self.name}: {dt:.3f}s {note}")

def _nan_guard(tag, x):
    if x is None: 
        return
    bad = torch.isnan(x).any() or torch.isinf(x).any()
    if bad: 
        logger.warning(f"[NAN] {tag} has NaN/Inf!")

def _clamp_span(l_min, l_max, L):
    l_min = max(0, min(l_min, L-1))
    l_max = max(l_min+1, min(l_max, L))
    return l_min, l_max


torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


SUPPORT_LANGUAGES = {
    "en": 259,
    "de": 260,
    "fr": 262,
    "es": 284,
    "it": 285,
    "pt": 286,
    "pl": 294,
    "tr": 295,
    "ru": 267,
    "cs": 293,
    "nl": 297,
    "ar": 5022,
    "zh": 5023,
    "ja": 5412,
    "hu": 5753,
    "ko": 6152,
    "hi": 6680,
}

structure_pattern = re.compile(r"\[.*?\]")


def ensure_directory_exists(directory):
    directory = str(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def _frames_per_second(sample_rate: int) -> float:
    """Calculate frames per second for the latent space."""
    HOP = 512
    RATIO = 8
    return float(sample_rate) / (HOP * RATIO)


REPO_ID = "ACE-Step/ACE-Step-v1-3.5B"
REPO_ID_QUANT = REPO_ID + "-q4-K-M" # ??? update this i guess


# class ACEStepPipeline(DiffusionPipeline):
class ACEStepPipeline:
    def _latent_rms(self, x, eps=1e-6):
        # RMS over time axis only, averaged across batch/channels
        return float(torch.sqrt(torch.clamp((x * x).mean(dim=-1), eps)).mean().detach().cpu())

    def _latent_stats(self, x):
        m = float(x.mean().detach().cpu())
        s = float(x.std().detach().cpu())
        mx = float(x.max().detach().cpu())
        mn = float(x.min().detach().cpu())
        return m, s, mn, mx

    def _avg_time(self, x, k=5):
        B, C, H, T = x.shape
        y = x.to(torch.float32).view(B*C, H, T)
        y = torch.nn.functional.avg_pool1d(y, kernel_size=k, stride=1, padding=k//2)
        return y.view(B, C, H, T).to(x.dtype)

    def _match_rms_like(self, z, ref, eps=1e-6):
        def _rms(v):
            return torch.sqrt(torch.clamp((v*v).mean(dim=-1, keepdim=True), eps))
        return z * (_rms(ref) / (_rms(z) + eps))

    def _match_global_stats(self, z, ref, eps=1e-6):
        m_ref = ref.mean(dim=(1,2,3), keepdim=True)
        s_ref = ref.std(dim=(1,2,3), keepdim=True) + eps
        m_z   = z.mean(dim=(1,2,3), keepdim=True)
        s_z   = z.std(dim=(1,2,3), keepdim=True) + eps
        return (z - m_z) * (s_ref / s_z) + m_ref

    def _safe_cap(self, name, val, lo, hi):
        if val < lo:
            logger.warning(f"[CAP] {name} {val} < {lo} -> {lo}")
            return lo
        if val > hi:
            logger.warning(f"[CAP] {name} {val} > {hi} -> {hi}")
            return hi
        return val

    def __init__(
        self,
        checkpoint_dir=None,
        device_id=0,
        dtype="bfloat16",
        text_encoder_checkpoint_path=None,
        persistent_storage_path=None,
        torch_compile=False,
        cpu_offload=False,
        quantized=False,
        overlapped_decode=False,
        **kwargs,
    ):
        if not checkpoint_dir:
            if persistent_storage_path is None:
                checkpoint_dir = os.path.join(
                    os.path.expanduser("~"), ".cache/ace-step/checkpoints"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
            else:
                checkpoint_dir = os.path.join(persistent_storage_path, "checkpoints")
        ensure_directory_exists(checkpoint_dir)

        self.checkpoint_dir = checkpoint_dir
        self.lora_path = "none"
        self.lora_weight = 1
        device = (
            torch.device(f"cuda:{device_id}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if device.type == "cpu" and torch.backends.mps.is_available():
            device = torch.device("mps")
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        if device.type == "mps" and self.dtype == torch.bfloat16:
            self.dtype = torch.float16
        if device.type == "mps":
            self.dtype = torch.float32
        if 'ACE_PIPELINE_DTYPE' in os.environ and len(os.environ['ACE_PIPELINE_DTYPE']):
            self.dtype = getattr(torch, os.environ['ACE_PIPELINE_DTYPE'])
        self.device = device
        self.loaded = False
        self.torch_compile = torch_compile
        self.cpu_offload = cpu_offload
        self.quantized = quantized
        self.overlapped_decode = overlapped_decode
        self.debug = DebugCfg(enabled=False)  # flip this on when you want deep logs

    def cleanup_memory(self):
        """Clean up GPU and CPU memory to prevent VRAM overflow during multiple generations."""
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            # Log memory usage if in verbose mode
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        # Collect Python garbage
        import gc
        gc.collect()

    def get_checkpoint_path(self, checkpoint_dir, repo):
        checkpoint_dir_models = None
        
        if checkpoint_dir is not None:
            required_dirs = ["music_dcae_f8c8", "music_vocoder", "ace_step_transformer", "umt5-base"]
            all_dirs_exist = True
            for dir_name in required_dirs:
                dir_path = os.path.join(checkpoint_dir, dir_name)
                if not os.path.exists(dir_path):
                    all_dirs_exist = False
                    break
            
            if all_dirs_exist:
                logger.info(f"Load models from: {checkpoint_dir}")
                checkpoint_dir_models = checkpoint_dir
        
        if checkpoint_dir_models is None:
            if checkpoint_dir is None:
                logger.info(f"Download models from Hugging Face: {repo}")
                checkpoint_dir_models = snapshot_download(repo)
            else:
                logger.info(f"Download models from Hugging Face: {repo}, cache to: {checkpoint_dir}")
                checkpoint_dir_models = snapshot_download(repo, cache_dir=checkpoint_dir)
        return checkpoint_dir_models

    def load_checkpoint(self, checkpoint_dir=None, export_quantized_weights=False):
        checkpoint_dir = self.get_checkpoint_path(checkpoint_dir, REPO_ID)
        dcae_checkpoint_path = os.path.join(checkpoint_dir, "music_dcae_f8c8")
        vocoder_checkpoint_path = os.path.join(checkpoint_dir, "music_vocoder")
        ace_step_checkpoint_path = os.path.join(checkpoint_dir, "ace_step_transformer")
        text_encoder_checkpoint_path = os.path.join(checkpoint_dir, "umt5-base")

        self.ace_step_transformer = ACEStepTransformer2DModel.from_pretrained(
            ace_step_checkpoint_path, torch_dtype=self.dtype
        )
        # self.ace_step_transformer.to(self.device).eval().to(self.dtype)
        if self.cpu_offload:
            self.ace_step_transformer = (
                self.ace_step_transformer.to("cpu").eval().to(self.dtype)
            )
        else:
            self.ace_step_transformer = (
                self.ace_step_transformer.to(self.device).eval().to(self.dtype)
            )
        if self.torch_compile and self.device.type == "cuda" and not self.quantized:
            self.ace_step_transformer = torch.compile(self.ace_step_transformer)

        self.music_dcae = MusicDCAE(
            dcae_checkpoint_path=dcae_checkpoint_path,
            vocoder_checkpoint_path=vocoder_checkpoint_path,
        )
        # self.music_dcae.to(self.device).eval().to(self.dtype)
        if self.cpu_offload:  # might be redundant
            self.music_dcae = self.music_dcae.to("cpu").eval().to(self.dtype)
        else:
            self.music_dcae = self.music_dcae.to(self.device).eval().to(self.dtype)
        if self.torch_compile and self.device.type == "cuda" and not self.quantized:
            self.music_dcae = torch.compile(self.music_dcae)

        lang_segment = LangSegment()
        lang_segment.setfilters(language_filters.default)
        self.lang_segment = lang_segment
        self.lyric_tokenizer = VoiceBpeTokenizer()

        text_encoder_model = UMT5EncoderModel.from_pretrained(
            text_encoder_checkpoint_path, torch_dtype=self.dtype
        ).eval()
        # text_encoder_model = text_encoder_model.to(self.device).to(self.dtype)
        if self.cpu_offload:
            text_encoder_model = text_encoder_model.to("cpu").eval().to(self.dtype)
        else:
            text_encoder_model = text_encoder_model.to(self.device).eval().to(self.dtype)
        text_encoder_model.requires_grad_(False)
        self.text_encoder_model = text_encoder_model
        if self.torch_compile and self.device.type == "cuda" and not self.quantized:
            self.text_encoder_model = torch.compile(self.text_encoder_model)

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_checkpoint_path
        )
        self.loaded = True

        # compile
        if self.torch_compile:
            if export_quantized_weights:
                from torch.ao.quantization import (
                    quantize_,
                    Int4WeightOnlyConfig,
                )

                group_size = 128
                use_hqq = True
                quantize_(
                    self.ace_step_transformer,
                    Int4WeightOnlyConfig(group_size=group_size, use_hqq=use_hqq),
                )
                quantize_(
                    self.text_encoder_model,
                    Int4WeightOnlyConfig(group_size=group_size, use_hqq=use_hqq),
                )

                # save quantized weights
                torch.save(
                    self.ace_step_transformer.state_dict(),
                    os.path.join(
                        ace_step_checkpoint_path, "diffusion_pytorch_model_int4wo.bin"
                    ),
                )
                print(
                    "Quantized Weights Saved to: ",
                    os.path.join(
                        ace_step_checkpoint_path, "diffusion_pytorch_model_int4wo.bin"
                    ),
                )
                torch.save(
                    self.text_encoder_model.state_dict(),
                    os.path.join(text_encoder_checkpoint_path, "pytorch_model_int4wo.bin"),
                )
                print(
                    "Quantized Weights Saved to: ",
                    os.path.join(text_encoder_checkpoint_path, "pytorch_model_int4wo.bin"),
                )

    def load_quantized_checkpoint(self, checkpoint_dir=None):
        checkpoint_dir = self.get_checkpoint_path(checkpoint_dir, REPO_ID_QUANT)
        dcae_checkpoint_path = os.path.join(checkpoint_dir, "music_dcae_f8c8")
        vocoder_checkpoint_path = os.path.join(checkpoint_dir, "music_vocoder")
        ace_step_checkpoint_path = os.path.join(checkpoint_dir, "ace_step_transformer")
        text_encoder_checkpoint_path = os.path.join(checkpoint_dir, "umt5-base")

        target = "cpu" if self.cpu_offload else self.device
        
        self.music_dcae = MusicDCAE(
            dcae_checkpoint_path=dcae_checkpoint_path,
            vocoder_checkpoint_path=vocoder_checkpoint_path,
        )
        self.music_dcae.eval().to(self.dtype).to(target)
        if self.torch_compile and self.device.type == "cuda" and not self.quantized:
            self.music_dcae = torch.compile(self.music_dcae)

        self.ace_step_transformer = ACEStepTransformer2DModel.from_pretrained(ace_step_checkpoint_path)
        self.ace_step_transformer.eval().to(self.dtype).to(target)
        if self.torch_compile and self.device.type == "cuda" and not self.quantized:
            self.ace_step_transformer = torch.compile(self.ace_step_transformer)
        self.ace_step_transformer.load_state_dict(
            torch.load(
                os.path.join(ace_step_checkpoint_path, "diffusion_pytorch_model_int4wo.bin"),
                map_location=target,
            ),assign=True
        )
        self.ace_step_transformer.torchao_quantized = True

        self.text_encoder_model = UMT5EncoderModel.from_pretrained(text_encoder_checkpoint_path)
        self.text_encoder_model.eval().to(self.dtype).to(target)
        if self.torch_compile and self.device.type == "cuda" and not self.quantized:
            self.text_encoder_model = torch.compile(self.text_encoder_model)
        self.text_encoder_model.load_state_dict(
            torch.load(
                os.path.join(text_encoder_checkpoint_path, "pytorch_model_int4wo.bin"),
                map_location=target,
            ), assign=True
        )
        self.text_encoder_model.torchao_quantized = True

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_checkpoint_path
        )

        lang_segment = LangSegment()
        lang_segment.setfilters(language_filters.default)
        self.lang_segment = lang_segment
        self.lyric_tokenizer = VoiceBpeTokenizer()

        self.loaded = True

    @cpu_offload("text_encoder_model")
    def get_text_embeddings(self, texts, text_max_length=256):
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.text_encoder_model.device != self.device:
            self.text_encoder_model.to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        return last_hidden_states, attention_mask

    @cpu_offload("text_encoder_model")
    def get_text_embeddings_null(
        self, texts, text_max_length=256, tau=0.01, l_min=8, l_max=10
    ):
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.text_encoder_model.device != self.device:
            self.text_encoder_model.to(self.device)

        def forward_with_temperature(inputs, tau=0.01, l_min=8, l_max=10):
            handlers = []

            def hook(module, input, output):
                output[:] *= tau
                return output

            L = len(self.text_encoder_model.encoder.block)
            lm, lx = _clamp_span(l_min, l_max, L)
            for i in range(lm, lx):
                handler = (
                    self.text_encoder_model.encoder.block[i]
                    .layer[0]
                    .SelfAttention.q.register_forward_hook(hook)
                )
                handlers.append(handler)

            with torch.no_grad():
                outputs = self.text_encoder_model(**inputs)
                last_hidden_states = outputs.last_hidden_state

            for hook in handlers:
                hook.remove()

            return last_hidden_states

        last_hidden_states = forward_with_temperature(inputs, tau, l_min, l_max)
        return last_hidden_states

    def set_seeds(self, batch_size, manual_seeds=None):
        processed_input_seeds = None
        if manual_seeds is not None:
            if isinstance(manual_seeds, str):
                if "," in manual_seeds:
                    processed_input_seeds = list(map(int, manual_seeds.split(",")))
                elif manual_seeds.isdigit():
                    processed_input_seeds = int(manual_seeds)
            elif isinstance(manual_seeds, list) and all(
                isinstance(s, int) for s in manual_seeds
            ):
                if len(manual_seeds) > 0:
                    processed_input_seeds = list(manual_seeds)
            elif isinstance(manual_seeds, int):
                processed_input_seeds = manual_seeds
        random_generators = [
            torch.Generator(device=self.device) for _ in range(batch_size)
        ]
        actual_seeds = []
        for i in range(batch_size):
            current_seed_for_generator = None
            if processed_input_seeds is None:
                current_seed_for_generator = torch.randint(0, 2**32, (1,)).item()
            elif isinstance(processed_input_seeds, int):
                current_seed_for_generator = processed_input_seeds
            elif isinstance(processed_input_seeds, list):
                if i < len(processed_input_seeds):
                    current_seed_for_generator = processed_input_seeds[i]
                else:
                    current_seed_for_generator = processed_input_seeds[-1]
            if current_seed_for_generator is None:
                current_seed_for_generator = torch.randint(0, 2**32, (1,)).item()
            random_generators[i].manual_seed(current_seed_for_generator)
            actual_seeds.append(current_seed_for_generator)
        return random_generators, actual_seeds

    def get_lang(self, text):
        language = "en"
        try:
            _ = self.lang_segment.getTexts(text)
            langCounts = self.lang_segment.getCounts()
            language = langCounts[0][0]
            if len(langCounts) > 1 and language == "en":
                language = langCounts[1][0]
        except Exception as err:
            language = "en"
        return language

    def tokenize_lyrics(self, lyrics, debug=False):
        lines = lyrics.split("\n")
        lyric_token_idx = [261]
        for line in lines:
            line = line.strip()
            if not line:
                lyric_token_idx += [2]
                continue

            lang = self.get_lang(line)

            if lang not in SUPPORT_LANGUAGES:
                lang = "en"
            if "zh" in lang:
                lang = "zh"
            if "spa" in lang:
                lang = "es"

            try:
                if structure_pattern.match(line):
                    token_idx = self.lyric_tokenizer.encode(line, "en")
                else:
                    token_idx = self.lyric_tokenizer.encode(line, lang)
                if debug:
                    toks = self.lyric_tokenizer.batch_decode(
                        [[tok_id] for tok_id in token_idx]
                    )
                    logger.info(f"debbug {line} --> {lang} --> {toks}")
                lyric_token_idx = lyric_token_idx + token_idx + [2]
            except Exception as e:
                print("tokenize error", e, "for line", line, "major_language", lang)
        return lyric_token_idx

    @cpu_offload("ace_step_transformer")
    def calc_v(
        self,
        zt_src,
        zt_tar,
        t,
        encoder_text_hidden_states,
        text_attention_mask,
        target_encoder_text_hidden_states,
        target_text_attention_mask,
        speaker_embds,
        target_speaker_embeds,
        lyric_token_ids,
        lyric_mask,
        target_lyric_token_ids,
        target_lyric_mask,
        do_classifier_free_guidance=False,
        guidance_scale=1.0,
        target_guidance_scale=1.0,
        cfg_type="apg",
        attention_mask=None,
        momentum_buffer=None,
        momentum_buffer_tar=None,
        return_src_pred=True,
    ):
        noise_pred_src = None
        if return_src_pred:
            src_latent_model_input = (
                torch.cat([zt_src, zt_src]) if do_classifier_free_guidance else zt_src
            )
            timestep = t.expand(src_latent_model_input.shape[0])
            # source
            noise_pred_src = self.ace_step_transformer(
                hidden_states=src_latent_model_input,
                attention_mask=attention_mask,
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embeds=speaker_embds,
                lyric_token_idx=lyric_token_ids,
                lyric_mask=lyric_mask,
                timestep=timestep,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_with_cond_src, noise_pred_uncond_src = noise_pred_src.chunk(
                    2
                )
                if cfg_type == "apg":
                    noise_pred_src = apg_forward(
                        pred_cond=noise_pred_with_cond_src,
                        pred_uncond=noise_pred_uncond_src,
                        guidance_scale=guidance_scale,
                        momentum_buffer=momentum_buffer,
                    )
                elif cfg_type == "cfg":
                    noise_pred_src = cfg_forward(
                        cond_output=noise_pred_with_cond_src,
                        uncond_output=noise_pred_uncond_src,
                        cfg_strength=guidance_scale,
                    )

        tar_latent_model_input = (
            torch.cat([zt_tar, zt_tar]) if do_classifier_free_guidance else zt_tar
        )
        timestep = t.expand(tar_latent_model_input.shape[0])
        # target
        noise_pred_tar = self.ace_step_transformer(
            hidden_states=tar_latent_model_input,
            attention_mask=attention_mask,
            encoder_text_hidden_states=target_encoder_text_hidden_states,
            text_attention_mask=target_text_attention_mask,
            speaker_embeds=target_speaker_embeds,
            lyric_token_idx=target_lyric_token_ids,
            lyric_mask=target_lyric_mask,
            timestep=timestep,
        ).sample

        if do_classifier_free_guidance:
            noise_pred_with_cond_tar, noise_pred_uncond_tar = noise_pred_tar.chunk(2)
            if cfg_type == "apg":
                noise_pred_tar = apg_forward(
                    pred_cond=noise_pred_with_cond_tar,
                    pred_uncond=noise_pred_uncond_tar,
                    guidance_scale=target_guidance_scale,
                    momentum_buffer=momentum_buffer_tar,
                )
            elif cfg_type == "cfg":
                noise_pred_tar = cfg_forward(
                    cond_output=noise_pred_with_cond_tar,
                    uncond_output=noise_pred_uncond_tar,
                    cfg_strength=target_guidance_scale,
                )
        return noise_pred_src, noise_pred_tar

    @torch.no_grad()
    def flowedit_diffusion_process(
        self,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        target_encoder_text_hidden_states,
        target_text_attention_mask,
        target_speaker_embeds,
        target_lyric_token_ids,
        target_lyric_mask,
        src_latents,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        n_min=0,
        n_max=1.0,
        n_avg=1,
        scheduler_type="euler",
    ):

        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        target_guidance_scale = guidance_scale
        bsz = encoder_text_hidden_states.shape[0]

        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        elif scheduler_type == "pingpong":
            scheduler = FlowMatchPingPongScheduler(num_train_timesteps=1000, shift=3.0)
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

        T_steps = infer_steps
        frame_length = src_latents.shape[-1]
        attention_mask = torch.ones(bsz, frame_length, device=self.device, dtype=torch.bool)

        timesteps, T_steps = retrieve_timesteps(
            scheduler, T_steps, self.device, timesteps=None
        )

        if do_classifier_free_guidance:
            attention_mask = torch.cat([attention_mask] * 2, dim=0)

            encoder_text_hidden_states = torch.cat(
                [
                    encoder_text_hidden_states,
                    torch.zeros_like(encoder_text_hidden_states),
                ],
                0,
            )
            text_attention_mask = torch.cat([text_attention_mask] * 2, dim=0)

            target_encoder_text_hidden_states = torch.cat(
                [
                    target_encoder_text_hidden_states,
                    torch.zeros_like(target_encoder_text_hidden_states),
                ],
                0,
            )
            target_text_attention_mask = torch.cat(
                [target_text_attention_mask] * 2, dim=0
            )

            speaker_embds = torch.cat(
                [speaker_embds, torch.zeros_like(speaker_embds)], 0
            )
            target_speaker_embeds = torch.cat(
                [target_speaker_embeds, torch.zeros_like(target_speaker_embeds)], 0
            )

            lyric_token_ids = torch.cat(
                [lyric_token_ids, torch.zeros_like(lyric_token_ids)], 0
            )
            lyric_mask = torch.cat([lyric_mask, torch.zeros_like(lyric_mask)], 0)

            target_lyric_token_ids = torch.cat(
                [target_lyric_token_ids, torch.zeros_like(target_lyric_token_ids)], 0
            )
            target_lyric_mask = torch.cat(
                [target_lyric_mask, torch.zeros_like(target_lyric_mask)], 0
            )

        momentum_buffer = MomentumBuffer()
        momentum_buffer_tar = MomentumBuffer()
        x_src = src_latents
        zt_edit = x_src.clone()
        xt_tar = None
        n_min = int(infer_steps * n_min)
        n_max = int(infer_steps * n_max)

        logger.info("flowedit start from {} to {}".format(n_min, n_max))

        use_bar = bool(self.debug.enabled)
        for i, t in tqdm(enumerate(timesteps), total=T_steps, disable=not use_bar):

            if i < n_min:
                continue

            t_i = t / 1000

            if i + 1 < len(timesteps):
                t_im1 = (timesteps[i + 1]) / 1000
            else:
                t_im1 = torch.zeros_like(t_i).to(self.device)

            if i < n_max:
                # Calculate the average of the V predictions
                V_delta_avg = torch.zeros_like(x_src)
                for k in range(n_avg):
                    fwd_noise = randn_tensor(
                        shape=x_src.shape,
                        generator=random_generators,
                        device=self.device,
                        dtype=self.dtype,
                    )

                    zt_src = (1 - t_i) * x_src + (t_i) * fwd_noise

                    zt_tar = zt_edit + zt_src - x_src

                    Vt_src, Vt_tar = self.calc_v(
                        zt_src=zt_src,
                        zt_tar=zt_tar,
                        t=t,
                        encoder_text_hidden_states=encoder_text_hidden_states,
                        text_attention_mask=text_attention_mask,
                        target_encoder_text_hidden_states=target_encoder_text_hidden_states,
                        target_text_attention_mask=target_text_attention_mask,
                        speaker_embds=speaker_embds,
                        target_speaker_embeds=target_speaker_embeds,
                        lyric_token_ids=lyric_token_ids,
                        lyric_mask=lyric_mask,
                        target_lyric_token_ids=target_lyric_token_ids,
                        target_lyric_mask=target_lyric_mask,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guidance_scale=guidance_scale,
                        target_guidance_scale=target_guidance_scale,
                        attention_mask=attention_mask,
                        momentum_buffer=momentum_buffer,
                    )
                    V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)  # - (hfg - 1) * (x_src)

                zt_edit = zt_edit.to(torch.float32)  # arbitrary, should be settable for compatibility
                if scheduler_type != "pingpong":
                    # propagate direct ODE
                    zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
                    zt_edit = zt_edit.to(self.dtype)
                else:
                    # propagate pingpong SDE
                    zt_edit_denoised = zt_edit - t_i * V_delta_avg
                    noise = torch.empty_like(zt_edit).normal_(generator=random_generators[0] if random_generators else None)
                    prev_sample = (1 - t_im1) * zt_edit_denoised + t_im1 * noise

            else:  # i >= T_steps-n_min # regular sampling for last n_min steps
                if i == n_max:
                    fwd_noise = randn_tensor(
                        shape=x_src.shape,
                        generator=random_generators,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    scheduler._init_step_index(t)
                    sigma = scheduler.sigmas[scheduler.step_index]
                    xt_src = sigma * fwd_noise + (1.0 - sigma) * x_src
                    xt_tar = zt_edit + xt_src - x_src

                _, Vt_tar = self.calc_v(
                    zt_src=None,
                    zt_tar=xt_tar,
                    t=t,
                    encoder_text_hidden_states=encoder_text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    target_encoder_text_hidden_states=target_encoder_text_hidden_states,
                    target_text_attention_mask=target_text_attention_mask,
                    speaker_embds=speaker_embds,
                    target_speaker_embeds=target_speaker_embeds,
                    lyric_token_ids=lyric_token_ids,
                    lyric_mask=lyric_mask,
                    target_lyric_token_ids=target_lyric_token_ids,
                    target_lyric_mask=target_lyric_mask,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guidance_scale=guidance_scale,
                    target_guidance_scale=target_guidance_scale,
                    attention_mask=attention_mask,
                    momentum_buffer_tar=momentum_buffer_tar,
                    return_src_pred=False,
                )

                xt_tar = xt_tar.to(torch.float32)
                if scheduler_type != "pingpong":
                    prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
                    prev_sample = prev_sample.to(self.dtype)
                    xt_tar = prev_sample
                else:
                    prev_sample = xt_tar - t_i * Vt_tar
                    noise = torch.empty_like(zt_edit).normal_(generator=random_generators[0] if random_generators else None)
                    prev_sample = (1 - t_im1) * prev_sample + t_im1 * noise
                    xt_tar = prev_sample

        target_latents = zt_edit if xt_tar is None else xt_tar
        return target_latents

    def add_latents_noise(
        self,
        gt_latents,
        sigma_max,
        noise,
        scheduler_type,
        infer_steps,
    ):

        bsz = gt_latents.shape[0]
        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
                sigma_max=sigma_max,
            )
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
                sigma_max=sigma_max,
            )
        elif scheduler_type == "pingpong":
            scheduler = FlowMatchPingPongScheduler(
                num_train_timesteps=1000,
                shift=3.0,
                sigma_max=sigma_max
            )

        infer_steps = max(1, int(round(sigma_max * infer_steps)))
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps=infer_steps,
            device=self.device,
            timesteps=None,
        )
        noisy_image = gt_latents * (1 - scheduler.sigma_max) + noise * scheduler.sigma_max
        first = float(timesteps[0]/1000)
        last = float(timesteps[-1]/1000)
        logger.info(f"{scheduler.sigma_min=} {scheduler.sigma_max=} steps={num_inference_steps} "
                    f"t=[{first:.4f}…{last:.4f}]")
        return noisy_image, timesteps, scheduler, num_inference_steps

    @cpu_offload("ace_step_transformer")
    @torch.no_grad()
    def text2music_diffusion_process(
        self,
        duration,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        omega_scale=10.0,
        scheduler_type="euler",
        cfg_type="apg",
        zero_steps=1,
        use_zero_init=True,
        guidance_interval=0.5,
        guidance_interval_decay=1.0,
        min_guidance_scale=3.0,
        oss_steps=[],
        encoder_text_hidden_states_null=None,
        use_erg_lyric=False,
        use_erg_diffusion=False,
        retake_random_generators=None,
        retake_variance=0.5,
        add_retake_noise=False,
        guidance_scale_text=0.0,
        guidance_scale_lyric=0.0,
        repaint_start=0,
        repaint_end=0,
        src_latents=None,
        audio2audio_enable=False,
        ref_audio_strength=0.5,
        ref_latents=None,
        extend_strength=0.7,
        sample_rate=48000,
        # ---- extend bootstrap controls ----
        extend_bootstrap=True,
        extend_bootstrap_method="a2a",
        extend_bootstrap_strength=0.6,
        seam_seconds=0.75,
        extend_pad_mode="boot_only",
        extend_tile_only=False,
        extend_edge_bootstrap_only=False,   # <--- NEW
        extend_tile_then_repaint: bool = False,
        # ---- extend bootstrap tuning (new) ----
        extend_bootstrap_edge_sec: float = 2.0,          # how many seconds of edge context to tile from
        extend_bootstrap_sigma_max: Optional[float] = None, # override the sigma_max used inside bootstrap; None -> derive from extend_bootstrap_strength
        extend_bootstrap_noise_mode: str = "zeros",      # "zeros" | "matched" | "gauss"
    ):
        
        # Mutual exclusion guards
        if extend_tile_only:
            extend_bootstrap = False
            extend_edge_bootstrap_only = False

        if self.debug.enabled:
            logger.info(f"[PATH] enter text2music_diffusion_process "
                        f"is_extend={(src_latents is not None and (repaint_start<0 or repaint_end> (src_latents.shape[-1]/_frames_per_second(44100))))} "
                        f"add_retake_noise={add_retake_noise}")

        logger.info(
            "cfg_type: {}, guidance_scale: {}, omega_scale: {}".format(
                cfg_type, guidance_scale, omega_scale
            )
        )
        
        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        do_double_condition_guidance = False
        if (
            guidance_scale_text is not None
            and guidance_scale_text > 1.0
            and guidance_scale_lyric is not None
            and guidance_scale_lyric > 1.0
        ):
            do_double_condition_guidance = True
            logger.info(
                "do_double_condition_guidance: {}, guidance_scale_text: {}, guidance_scale_lyric: {}".format(
                    do_double_condition_guidance,
                    guidance_scale_text,
                    guidance_scale_lyric,
                )
            )

        bsz = encoder_text_hidden_states.shape[0]

        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )
        elif scheduler_type == "pingpong":
            scheduler = FlowMatchPingPongScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )

        if self.debug.enabled:
            logger.info(f"[PATH] scheduler={scheduler_type}")

        # Use the correct sample rate for frame calculations
        # If we have source latents, use 44.1kHz (MusicDCAE's native rate)
        # Otherwise use the user-provided sample_rate
        if src_latents is not None:
            # Use MusicDCAE's native sample rate (44.1kHz) for frame calculations
            fps = _frames_per_second(44100)
            logger.info(f"Using 44.1kHz for frame calculations (MusicDCAE native rate): fps={fps:.6f}")
        else:
            fps = _frames_per_second(sample_rate)
            logger.info(f"Using {sample_rate}Hz for frame calculations: fps={fps:.6f}")
            
        frame_length = int(duration * fps)
        if src_latents is not None:
            frame_length = src_latents.shape[-1]
            
        logger.info(f"[DBG] fps={fps:.6f}  infer_steps={infer_steps}  guidance_scale={guidance_scale}  min_cfg={min_guidance_scale} cfg_type={cfg_type} scheduler={type(scheduler).__name__}")
        logger.info(f"[DBG] initial frame_length={frame_length}  src_latents={'yes' if src_latents is not None else 'no'}  ref_latents={'yes' if ref_latents is not None else 'no'}")
        
        if ref_latents is not None:
            frame_length = ref_latents.shape[-1]

        if len(oss_steps) > 0:
            infer_steps = max(oss_steps)
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler,
                num_inference_steps=infer_steps,
                device=self.device,
                timesteps=None,
            )
            new_timesteps = torch.zeros(len(oss_steps), dtype=self.dtype, device=self.device)
            for idx in range(len(oss_steps)):
                new_timesteps[idx] = timesteps[oss_steps[idx] - 1]
            num_inference_steps = len(oss_steps)
            sigmas = (new_timesteps / 1000).float().cpu().numpy()
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler,
                num_inference_steps=num_inference_steps,
                device=self.device,
                sigmas=sigmas,
            )
            logger.info(f"oss_steps={oss_steps} steps={num_inference_steps} "
                        f"t=[{float(timesteps[0])/1000:.4f}…{float(timesteps[-1])/1000:.4f}]")
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler,
                num_inference_steps=infer_steps,
                device=self.device,
                timesteps=None,
            )

        target_latents = randn_tensor(
            shape=(bsz, 8, 16, frame_length),
            generator=random_generators,
            device=self.device,
            dtype=self.dtype,
        )
        _nan_guard("target_latents@init", target_latents)

        is_repaint = False
        is_extend = False

        if add_retake_noise:
            # Keep a scalar copy for scheduling math (fraction in [0,1])
            retake_frac = float(retake_variance)
            n_min = int(infer_steps * (1 - retake_frac))
            # ensure the '== n_min' branch is reachable
            n_min = max(1, min(n_min, infer_steps - 1))
            
            def _choose_n_min_by_sigma(target_sigma=0.20):
                # find earliest i where scheduler sigma <= target_sigma
                try:
                    # ensure scheduler indices initialized
                    _ = scheduler.sigmas
                    for i, t in enumerate(timesteps):
                        scheduler._init_step_index(t)
                        sig = float(scheduler.sigmas[scheduler.step_index].detach().cpu())
                        if sig <= target_sigma:
                            return max(1, i)  # keep >=1
                except Exception:
                    pass
                return None

            # Override n_min to ensure low sigma repaint
            late = _choose_n_min_by_sigma(0.20)
            if late is not None and late > n_min:
                logger.info(f"[DBG] overriding n_min {n_min} -> {late} to ensure low sigma repaint")
                n_min = late
            
            # Angle (tensor) used for mixing latents
            retake_variance = (
                torch.tensor(retake_frac * math.pi / 2).to(self.device).to(self.dtype)
            )
            retake_latents = randn_tensor(
                shape=(bsz, 8, 16, frame_length),
                generator=retake_random_generators,
                device=self.device,
                dtype=self.dtype,
            )
            def sec_to_frames(sec: float) -> int:
                return int(round(sec * fps))
            repaint_start_frame = sec_to_frames(repaint_start)
            repaint_end_frame = sec_to_frames(repaint_end)
            
            # Log extend/repaint setup details
            logger.info(f"repaint_start={repaint_start}, repaint_end={repaint_end}")
            logger.info(f"src_len_frames={src_latents.shape[-1] if src_latents is not None else None}, frame_length={frame_length}")
            
            def _sec(frames): 
                return frames / max(1.0, fps)
            
            logger.info(f"[DBG] repaint_start_frame={repaint_start_frame} ({_sec(repaint_start_frame):.3f}s)  repaint_end_frame={repaint_end_frame} ({_sec(repaint_end_frame):.3f}s)")
            
        # Only if we're repainting/retaking/extending (i.e., add_retake_noise) AND we have src_latents
        if add_retake_noise and (src_latents is not None):
            # Clamp to actual src length and guarantee ordering
            src_len = src_latents.shape[-1]
            
            if not ((repaint_start < 0) or (repaint_end > src_len / fps)):
                # repaint-only behavior: clamp to within the clip
                repaint_start_frame = max(0, min(repaint_start_frame, src_len))
                repaint_end_frame = max(0, min(repaint_end_frame, src_len))
            else:
                # extend behavior: allow overshoot on the right and negative on the left
                repaint_start_frame = max(-src_len, repaint_start_frame)  # allow negative
                # DO NOT clamp repaint_end_frame to src_len here
                repaint_end_frame = max(0, repaint_end_frame)  # only ensure non-negative
            
            if repaint_end_frame < repaint_start_frame:
                repaint_end_frame = repaint_start_frame

            # ↓↓↓ DEDENT everything from here ↓↓↓
            x0 = src_latents
            # retake
            is_repaint = repaint_end_frame - repaint_start_frame != frame_length

            is_extend = (repaint_start_frame < 0) or (repaint_end_frame > frame_length)
            if is_extend:
                is_repaint = True

            if self.debug.enabled:
                logger.info(f"[PATH] repaint? {is_repaint}  extend? {is_extend} "
                            f"frames(start={repaint_start_frame}, end={repaint_end_frame}, src_len={src_len if src_latents is not None else None})")

            # TODO: train a mask aware repainting controlnet
            # to make sure mean = 0, std = 1
            if not is_repaint:
                target_latents = (
                    torch.cos(retake_variance) * target_latents
                    + torch.sin(retake_variance) * retake_latents
                )
            elif not is_extend:
                # if repaint_end_frame
                repaint_mask = torch.zeros(
                    (bsz, 8, 16, frame_length), device=self.device, dtype=self.dtype
                )
                repaint_mask[:, :, :, repaint_start_frame:repaint_end_frame] = 1.0
                repaint_noise = (
                    torch.cos(retake_variance) * target_latents
                    + torch.sin(retake_variance) * retake_latents
                )
                repaint_noise = torch.where(
                    repaint_mask == 1.0, repaint_noise, target_latents
                )
                zt_edit = x0.clone()
                z0 = repaint_noise
            elif is_extend:
                logger.info("[PATH] TILE-ONLY branch entered")
                _tstats("x0(orig)", src_latents)
                
                gt_latents = src_latents
                src_len = gt_latents.shape[-1]
                max_infer_fame_length = int(240 * fps)

                # ---- RIGHT-ONLY EXTEND ----
                # Ignore negative repaint_start for extension; do not pad the left at all.
                # Compute desired extension purely from repaint_end.
                desired_end_f = max(src_len, repaint_end_frame)  # never earlier than src end
                right_pad = max(0, desired_end_f - src_len)
                new_len   = src_len + right_pad

                # Cap to model length
                if new_len > max_infer_fame_length:
                    right_pad = max_infer_fame_length - src_len
                    new_len   = src_len + right_pad

                # Single-sided pad (time is last dim)
                if right_pad > 0:
                    gt_latents = torch.nn.functional.pad(gt_latents, (0, right_pad), "constant", 0)

                # Update frame_length and repaint frames (only right region is editable)
                frame_length = new_len
                repaint_start_frame = src_len               # start repaint exactly at seam
                repaint_end_frame   = src_len + right_pad   # repaint only the extension

                # Logs
                f2s = lambda f: f / fps
                logger.info(f"[EXTEND-R] src={src_len}f ({f2s(src_len):.2f}s) right_pad={right_pad}f ({f2s(right_pad):.2f}s) new_len={new_len}f ({f2s(new_len):.2f}s)")

                # Build repaint mask: 0 on original clip, 1 on padded tail, with a short taper at the seam
                x0 = gt_latents
                repaint_mask = torch.zeros_like(gt_latents)
                if right_pad > 0:
                    repaint_mask[..., -right_pad:] = 1.0

                # ---- TILE-ONLY SEED FOR THE TAIL (no bootstrap) ----
                # Use last EDGE seconds as a motif; repeat to fill the extension
                def _avg_time(x, k=5):
                    B, C, H, T = x.shape
                    y = x.to(torch.float32).view(B*C, H, T)
                    y = torch.nn.functional.avg_pool1d(y, kernel_size=k, stride=1, padding=k//2)
                    return y.view(B, C, H, T).to(x.dtype)

                def _tile(edge_chunk, need):
                    reps = (need + edge_chunk.shape[-1] - 1) // edge_chunk.shape[-1]
                    return edge_chunk.repeat(1,1,1,reps)[..., :need]

                EDGE = max(64, int(round(float(extend_bootstrap_edge_sec) * fps)))
                mid  = x0[..., :src_len]  # original clip unchanged

                right_seed = None
                if right_pad > 0:
                    # take last EDGE frames (no flip), smooth a little, tile
                    right_edge = _avg_time(x0[..., src_len-EDGE:src_len], k=7) if src_len >= EDGE else _avg_time(x0[..., :src_len], k=7)
                    right_seed = _tile(right_edge, right_pad)

                    # cosine cross-fade at the seam to avoid a step
                    R = right_pad
                    fade = torch.linspace(0.0, 1.0, steps=min(int(round(seam_seconds * fps)), R), device=self.device, dtype=self.dtype)
                    # Align fade to the front of the pad region
                    fade_len = fade.numel()
                    if fade_len > 0:
                        # blend last fade_len frames of mid into first fade_len of right_seed
                        cross_A = mid[..., -fade_len:]
                        cross_B = right_seed[..., :fade_len]
                        cross   = (1.0 - fade).view(1,1,1,-1) * cross_A + fade.view(1,1,1,-1) * cross_B
                        right_seed = torch.cat([cross, right_seed[..., fade_len:]], dim=-1)

                # Concatenate: original mid + tiled tail
                parts = [mid]
                if right_seed is not None:
                    parts.append(right_seed)
                z0 = torch.cat(parts, dim=-1)

                _tstats("tiled(before stats match)", z0)

                # RMS + global distribution match keeps vocoder stable
                z0 = self._match_rms_like(z0, x0)
                z0 = self._match_global_stats(z0, x0)

                _tstats("tiled(after stats match)", z0)

                target_latents = z0.clone()
                zt_edit = x0.clone()

                _tstats("x0(padded/ref)", x0)
                _tstats("z0(seed)", z0)
                _tstats("repaint_mask mean", repaint_mask.mean())
                logger.info(f"[MASK] repaint coverage: {(repaint_mask>0).float().mean().item():.2%}  ramp_sec={seam_seconds}")

                # soften the repaint mask at seam (on the right only)
                ramp_len = int(max(1, round(float(seam_seconds) * fps)))
                if right_pad > 0:
                    L = min(ramp_len, right_pad)
                    ramp = torch.linspace(0.0, 1.0, steps=L, device=self.device, dtype=self.dtype).view(1,1,1,L)
                    s = frame_length - right_pad
                    repaint_mask[..., s : s + L] = ramp

                is_repaint = True
                is_extend  = True
                logger.info(f"[TILE→REPAINT-R] right-only seed (EDGE={EDGE}) → repaint tail; left kept intact.")
                
                # Seam check before repaint starts
                if is_extend and right_pad > 0:
                    seam_L = min(int(round(seam_seconds*fps)), right_pad, 1024)
                    left_edge  = x0[..., src_len-seam_L:src_len]
                    right_edge = z0[..., src_len:src_len+seam_L]
                    diff = (left_edge - right_edge).abs().amax().item()
                    logger.info(f"[SEAM] max_abs_diff={diff:.6f} over {seam_L}f")
                
                # Optional latents snapshots
                if hasattr(self, "debug") and self.debug.enabled and self.debug.dump_latents_npz:
                    npz_name = f"dbg_latents_{self.debug.run_tag}_prepaint.npz"
                    try:
                        import numpy as np
                        np.savez_compressed(npz_name, x0=x0.detach().cpu().numpy(),
                                            z0=z0.detach().cpu().numpy(),
                                            mask=repaint_mask.detach().cpu().numpy())
                        logger.info(f"[DBG] wrote {npz_name}")
                    except Exception as e:
                        logger.warning(f"[DBG] latents npz failed: {e}")

        if audio2audio_enable and ref_latents is not None:
            logger.info(
                f"audio2audio_enable: {audio2audio_enable}, ref_latents: {ref_latents.shape}"
            )
            target_latents, timesteps, scheduler, num_inference_steps = self.add_latents_noise(
                gt_latents=ref_latents,
                sigma_max=(1-ref_audio_strength),
                noise=target_latents,
                scheduler_type=scheduler_type,
                infer_steps=infer_steps,
            )

        attention_mask = torch.ones(bsz, frame_length, device=self.device, dtype=torch.bool)

        # guidance interval
        start_idx = int(num_inference_steps * ((1 - guidance_interval) / 2))
        end_idx = int(num_inference_steps * (guidance_interval / 2 + 0.5))
        
        # For long extends we keep guidance active across the full schedule
        if is_extend:
            start_idx = 0
            end_idx = num_inference_steps
            guidance_interval_decay = 1.2 if guidance_interval_decay == 0.0 else guidance_interval_decay
            min_guidance_scale = max(6.0, float(min_guidance_scale))
            omega_scale = min(10.0, float(omega_scale))   # avoid too aggressive steps
            logger.info(f"[EXTEND-STAB] CFG in full range, min_cfg={min_guidance_scale}, decay={guidance_interval_decay}, omega={omega_scale}")
        
        logger.info(
            f"start_idx: {start_idx}, end_idx: {end_idx}, num_inference_steps: {num_inference_steps}"
        )

        momentum_buffer = MomentumBuffer()

        def forward_encoder_with_temperature(self, inputs, tau=0.01, l_min=4, l_max=6):
            handlers = []

            def hook(module, input, output):
                output[:] *= tau
                return output

            L = len(self.ace_step_transformer.lyric_encoder.encoders)
            lm, lx = _clamp_span(l_min, l_max, L)
            for i in range(lm, lx):
                handler = self.ace_step_transformer.lyric_encoder.encoders[
                    i
                ].self_attn.linear_q.register_forward_hook(hook)
                handlers.append(handler)

            encoder_hidden_states, encoder_hidden_mask = (
                self.ace_step_transformer.encode(**inputs)
            )

            for hook in handlers:
                hook.remove()

            return encoder_hidden_states

        # P(speaker, text, lyric)
        encoder_hidden_states, encoder_hidden_mask = self.ace_step_transformer.encode(
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
        )

        if use_erg_lyric:
            # P(null_speaker, text_weaker, lyric_weaker)
            encoder_hidden_states_null = forward_encoder_with_temperature(
                self,
                inputs={
                    "encoder_text_hidden_states": (
                        encoder_text_hidden_states_null
                        if encoder_text_hidden_states_null is not None
                        else torch.zeros_like(encoder_text_hidden_states)
                    ),
                    "text_attention_mask": text_attention_mask,
                    "speaker_embeds": torch.zeros_like(speaker_embds),
                    "lyric_token_idx": lyric_token_ids,
                    "lyric_mask": lyric_mask,
                },
            )
        else:
            # P(null_speaker, null_text, null_lyric)
            encoder_hidden_states_null, _ = self.ace_step_transformer.encode(
                torch.zeros_like(encoder_text_hidden_states),
                text_attention_mask,
                torch.zeros_like(speaker_embds),
                torch.zeros_like(lyric_token_ids),
                lyric_mask,
            )

        encoder_hidden_states_no_lyric = None
        if do_double_condition_guidance:
            # P(null_speaker, text, lyric_weaker)
            if use_erg_lyric:
                encoder_hidden_states_no_lyric = forward_encoder_with_temperature(
                    self,
                    inputs={
                        "encoder_text_hidden_states": encoder_text_hidden_states,
                        "text_attention_mask": text_attention_mask,
                        "speaker_embeds": torch.zeros_like(speaker_embds),
                        "lyric_token_idx": lyric_token_ids,
                        "lyric_mask": lyric_mask,
                    },
                )
            # P(null_speaker, text, no_lyric)
            else:
                encoder_hidden_states_no_lyric, _ = self.ace_step_transformer.encode(
                    encoder_text_hidden_states,
                    text_attention_mask,
                    torch.zeros_like(speaker_embds),
                    torch.zeros_like(lyric_token_ids),
                    lyric_mask,
                )

        def forward_diffusion_with_temperature(
            self, hidden_states, timestep, inputs, tau=0.01, l_min=15, l_max=20
        ):
            handlers = []

            def hook(module, input, output):
                output[:] *= tau
                return output

            L = len(self.ace_step_transformer.transformer_blocks)
            lm, lx = _clamp_span(l_min, l_max, L)
            for i in range(lm, lx):
                handler = self.ace_step_transformer.transformer_blocks[
                    i
                ].attn.to_q.register_forward_hook(hook)
                handlers.append(handler)
                handler = self.ace_step_transformer.transformer_blocks[
                    i
                ].cross_attn.to_q.register_forward_hook(hook)
                handlers.append(handler)

            sample = self.ace_step_transformer.decode(
                hidden_states=hidden_states, timestep=timestep, **inputs
            ).sample

            for hook in handlers:
                hook.remove()

            return sample

        # Final shape guard: ensure z0 and target_latents match x0 along last dim
        # Only run if we're in repaint/extend mode and have the required variables
        if add_retake_noise and (src_latents is not None) and is_repaint:
            need = x0.shape[-1]

            if z0.shape[-1] > need:
                z0 = z0[..., :need]
            elif z0.shape[-1] < need:
                z0 = torch.nn.functional.pad(z0, (0, need - z0.shape[-1]))

            if target_latents.shape[-1] > need:
                target_latents = target_latents[..., :need]
            elif target_latents.shape[-1] < need:
                target_latents = torch.nn.functional.pad(target_latents, (0, need - target_latents.shape[-1]))
                
            logger.info(f"[DBG] shape guard enforced: target_latents={tuple(target_latents.shape)} z0={tuple(z0.shape)} x0={tuple(x0.shape)} need={need}")

        # Ensure repaint actually starts inside the schedule and at low noise
        if add_retake_noise:
            n_min = self._safe_cap("n_min", n_min, 1, num_inference_steps - 2)

            # Log sigma around n_min
            try:
                scheduler._init_step_index(timesteps[n_min])
                sig_here = float(scheduler.sigmas[scheduler.step_index].detach().cpu())
            except Exception:
                sig_here = float(timesteps[n_min] / 1000)
            logger.info(f"[REPAINT] n_min={n_min}/{num_inference_steps}  est_sigma={sig_here:.4f}  start_idx={start_idx} end_idx={end_idx}")
            logger.info(f"[PATH] repaint n_min={n_min} of {num_inference_steps} "
                        f"(retake_variance={'tensor' if isinstance(retake_variance, torch.Tensor) else float(retake_variance)})")

        use_bar = bool(self.debug.enabled)
        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps, disable=not use_bar):
            if self.debug.enabled and (i % max(1, self.debug.step_every) == 0):
                # short pulse logs
                logger.info(f"[STEP] i={i}/{num_inference_steps} t={float(t)/1000:.4f} "
                            f"in_guidance={start_idx<=i<end_idx} repaint={is_repaint and i>=n_min}")
                
                # Scheduler sigma probe
                try:
                    scheduler._init_step_index(t)
                    sig = float(scheduler.sigmas[scheduler.step_index])
                    logger.info(f"[SIG] i={i} sigma={sig:.5f}")
                except Exception:
                    pass

            if add_retake_noise and (src_latents is not None) and is_repaint:
                if i < n_min:
                    continue
                elif i == n_min:
                    t_i = t / 1000
                    zt_src = (1 - t_i) * x0 + (t_i) * z0
                    target_latents = zt_edit + zt_src - x0
                    logger.info(f"repaint start from {n_min} add {t_i} level of noise")
                    
                    # Estimate sigma at this t (scheduler must be initialized)
                    try:
                        scheduler._init_step_index(t)
                        sigma_here = float(scheduler.sigmas[scheduler.step_index].detach().cpu())
                    except Exception:
                        sigma_here = float(t_i)
                    logger.info(f"[DBG] repaint n_min={n_min}/{num_inference_steps}  sigma≈{sigma_here:.4f}  retake_frac={float(retake_variance) if isinstance(retake_variance, torch.Tensor)==False else 'tensor'}")
                    
                    # Key step stats
                    if i in {n_min, n_min+1, end_idx-1, end_idx}:
                        _tstats("latents@step", target_latents)

            # expand the latents if we are doing classifier free guidance
            latents = target_latents

            is_in_guidance_interval = start_idx <= i < end_idx
            if is_in_guidance_interval and do_classifier_free_guidance:
                # compute current guidance scale
                if guidance_interval_decay > 0:
                    # Linearly interpolate to calculate the current guidance scale
                    den = max(1, (end_idx - start_idx - 1))
                    progress = (i - start_idx) / den  # 归一化到[0,1]
                    current_guidance_scale = (
                        guidance_scale
                        - (guidance_scale - min_guidance_scale)
                        * progress
                        * guidance_interval_decay
                    )
                else:
                    current_guidance_scale = guidance_scale

                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])
                output_length = latent_model_input.shape[-1]
                # P(x|speaker, text, lyric)
                noise_pred_with_cond = self.ace_step_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=output_length,
                    timestep=timestep,
                ).sample

                noise_pred_with_only_text_cond = None
                if (
                    do_double_condition_guidance
                    and encoder_hidden_states_no_lyric is not None
                ):
                    noise_pred_with_only_text_cond = self.ace_step_transformer.decode(
                        hidden_states=latent_model_input,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states_no_lyric,
                        encoder_hidden_mask=encoder_hidden_mask,
                        output_length=output_length,
                        timestep=timestep,
                    ).sample

                if use_erg_diffusion:
                    noise_pred_uncond = forward_diffusion_with_temperature(
                        self,
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        inputs={
                            "encoder_hidden_states": encoder_hidden_states_null,
                            "encoder_hidden_mask": encoder_hidden_mask,
                            "output_length": output_length,
                            "attention_mask": attention_mask,
                        },
                    )
                else:
                    noise_pred_uncond = self.ace_step_transformer.decode(
                        hidden_states=latent_model_input,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states_null,
                        encoder_hidden_mask=encoder_hidden_mask,
                        output_length=output_length,
                        timestep=timestep,
                    ).sample

                if (
                    do_double_condition_guidance
                    and noise_pred_with_only_text_cond is not None
                ):
                    noise_pred = cfg_double_condition_forward(
                        cond_output=noise_pred_with_cond,
                        uncond_output=noise_pred_uncond,
                        only_text_cond_output=noise_pred_with_only_text_cond,
                        guidance_scale_text=guidance_scale_text,
                        guidance_scale_lyric=guidance_scale_lyric,
                    )

                elif cfg_type == "apg":
                    noise_pred = apg_forward(
                        pred_cond=noise_pred_with_cond,
                        pred_uncond=noise_pred_uncond,
                        guidance_scale=current_guidance_scale,
                        momentum_buffer=momentum_buffer,
                    )
                elif cfg_type == "cfg":
                    noise_pred = cfg_forward(
                        cond_output=noise_pred_with_cond,
                        uncond_output=noise_pred_uncond,
                        cfg_strength=current_guidance_scale,
                    )
                elif cfg_type == "cfg_star":
                    noise_pred = cfg_zero_star(
                        noise_pred_with_cond=noise_pred_with_cond,
                        noise_pred_uncond=noise_pred_uncond,
                        guidance_scale=current_guidance_scale,
                        i=i,
                        zero_steps=zero_steps,
                        use_zero_init=use_zero_init,
                    )
                
                # NaN guard for noise prediction
                _nan_guard("noise_pred", noise_pred)
            else:
                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])
                noise_pred = self.ace_step_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=latent_model_input.shape[-1],
                    timestep=timestep,
                ).sample
                
                # NaN guard for noise prediction
                _nan_guard("noise_pred", noise_pred)

            if is_repaint and i >= n_min:
                t_i = t / 1000
                if i + 1 < len(timesteps):
                    t_im1 = (timesteps[i + 1]) / 1000
                else:
                    t_im1 = torch.zeros_like(t_i).to(self.device)
                target_latents = target_latents.to(torch.float32)
                prev_sample = target_latents + (t_im1 - t_i) * noise_pred
                prev_sample = prev_sample.to(self.dtype)
                target_latents = prev_sample
                zt_src = (1 - t_im1) * x0 + (t_im1) * z0
                
                # Mid-run probes
                if i in (n_min, n_min+1, end_idx-2, end_idx-1):
                    rms_prev = self._latent_rms(prev_sample)
                    m,s,_,_ = self._latent_stats(prev_sample)
                    logger.info(f"[STEP] i={i} t={(float(t)/1000):.4f} rms(prev)={rms_prev:.6f} std(prev)={s:.5f} mean(prev)={m:.5f}")
                
                if i in (n_min, n_min+1, end_idx-1, end_idx):
                    logger.info(f"[DBG] step={i}  t={float(t)/1000:.4f}  in_guidance={start_idx <= i < end_idx}  rms(prev)={_rms_t(prev_sample):.6f}")
                
                if is_repaint and i >= n_min and (i % max(1, self.debug.step_every) == 0):
                    _tstats("prev_sample", prev_sample)
                
                # Soft blend (mask is 1.0 deep in pad, tapers to 0.0 at the seam)
                target_latents = repaint_mask * target_latents + (1.0 - repaint_mask) * zt_src
            else:
                generator = (random_generators[0] if random_generators else None)
                target_latents = scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=target_latents,
                    return_dict=False,
                    omega=omega_scale,
                    generator=generator,
                )[0]

        # Note: Re-append code removed - we already trimmed by reducing left_pad/right_pad
        # so there's nothing left to re-append
        
        logger.info(f"[DBG] DONE: is_extend={is_extend}  repaint={(is_repaint)}  final_latents={tuple(target_latents.shape)}  rms={_rms_t(target_latents):.6f}")
        return target_latents

    @cpu_offload("music_dcae")
    def latents2audio(
        self,
        latents,
        target_wav_duration_second=30,
        sample_rate=48000,
        save_path=None,
        format="wav",
    ):
        output_audio_paths = []
        bs = latents.shape[0]
        pred_latents = latents
        with torch.no_grad():
            if self.overlapped_decode and target_wav_duration_second > 48:
                _, pred_wavs = self.music_dcae.decode_overlap(pred_latents, sr=sample_rate)
            else:
                _, pred_wavs = self.music_dcae.decode(pred_latents, sr=sample_rate)
        pred_wavs = [pred_wav.cpu().float() for pred_wav in pred_wavs]
        
        # Audio insights after decode
        if hasattr(self, "debug") and self.debug.enabled:
            for bi, w in enumerate(pred_wavs):
                r = _rms(w)
                pk = float(w.abs().max())
                cf = (pk / (r + 1e-12)) if r > 0 else float("inf")  # crest factor
                dc = float(w.mean())
                logger.info(f"[AUDIO] batch={bi} rms={float(r):.6f} ({_db(r):.2f} dBFS) "
                            f"peak={pk:.6f} ({_db(pk):.2f} dBFS) crest={cf:.2f} dc={dc:.6f}")

                # Spectral stats
                try:
                    spec = torch.stft(w, n_fft=2048, hop_length=512, return_complex=True).abs()  # (F,T)
                    freqs = torch.linspace(0, sample_rate/2, spec.size(0), device=spec.device)
                    p = spec + 1e-12
                    centroid = (p * freqs[:,None]).sum(dim=0) / p.sum(dim=0)                       # per frame
                    rolloff_idx = (p.cumsum(dim=0) / p.sum(dim=0)).ge(0.85).float().argmax(dim=0)  # 85% rolloff
                    rolloff = freqs[rolloff_idx].float().mean().item()
                    logger.info(f"[AUDIO] centroid_mean={centroid.mean().item():.1f}Hz "
                                f"rolloff85={rolloff:.1f}Hz")
                except Exception as e:
                    logger.warning(f"[AUDIO] spectral stats failed: {e}")

                if self.debug.dump_specs:
                    png = f"dbg_spec_{self.debug.run_tag}_b{bi}_sr{sample_rate}.png"
                    _mel_spectrogram_png(w.unsqueeze(0), sample_rate, png)
        
        use_bar = bool(self.debug.enabled)
        for i in tqdm(range(bs), disable=not use_bar):
            output_audio_path = self.save_wav_file(
                pred_wavs[i],
                i,
                save_path=save_path,
                sample_rate=sample_rate,
                format=format,
            )
            output_audio_paths.append(output_audio_path)
        return output_audio_paths

    def save_wav_file(
        self, target_wav, idx, save_path=None, sample_rate=48000, format="wav"
    ):
        if save_path is None:
            logger.warning("save_path is None, using default path ./outputs/")
            base_path = "./outputs"
            ensure_directory_exists(base_path)
            output_path_wav = (
                f"{base_path}/output_{time.strftime('%Y%m%d%H%M%S')}_{idx}."+format
            )
        else:
            dirpart = os.path.dirname(save_path)
            if dirpart:
                ensure_directory_exists(dirpart)
            if os.path.isdir(save_path):
                logger.info(f"Provided save_path '{save_path}' is a directory. Appending timestamped filename.")
                output_path_wav = os.path.join(save_path, f"output_{time.strftime('%Y%m%d%H%M%S')}_{idx}."+format)
            else:
                output_path_wav = save_path

        target_wav = target_wav.float()
        backend = "sox_io" if format.lower() == "ogg" else "soundfile"
        logger.info(f"Saving audio to {output_path_wav} using backend {backend}")
        torchaudio.save(
            output_path_wav, target_wav, sample_rate=sample_rate, format=format, backend=backend
        )
        return output_path_wav

    @cpu_offload("music_dcae")
    def infer_latents(self, input_audio_path):
        if input_audio_path is None:
            return None
        input_audio, sr = self.music_dcae.load_audio(input_audio_path)
        input_audio = input_audio.unsqueeze(0)
        input_audio = input_audio.to(device=self.device, dtype=self.dtype)
        latents, _ = self.music_dcae.encode(input_audio, sr=sr)
        return latents

    def load_lora(self, lora_name_or_path, lora_weight):
        if (lora_name_or_path != self.lora_path or lora_weight != self.lora_weight) and lora_name_or_path != "none":
            if not os.path.exists(lora_name_or_path):
                lora_download_path = snapshot_download(lora_name_or_path, cache_dir=self.checkpoint_dir)
            else:
                lora_download_path = lora_name_or_path
            if self.lora_path != "none":
                self.ace_step_transformer.unload_lora()
            self.ace_step_transformer.load_lora_adapter(os.path.join(lora_download_path, "pytorch_lora_weights.safetensors"), adapter_name="ace_step_lora", with_alpha=True, prefix=None)
            logger.info(f"Loading lora weights from: {lora_name_or_path} download path is: {lora_download_path} weight: {lora_weight}")
            set_weights_and_activate_adapters(self.ace_step_transformer, ["ace_step_lora"], [lora_weight])
            self.lora_path = lora_name_or_path
            self.lora_weight = lora_weight
        elif self.lora_path != "none" and lora_name_or_path == "none":
            logger.info("No lora weights to load.")
            self.ace_step_transformer.unload_lora()

    def __call__(
        self,
        format: str = "wav",
        audio_duration: float = 60.0,
        prompt: str = None,
        lyrics: str = None,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        manual_seeds: list = None,
        guidance_interval: float = 0.5,
        guidance_interval_decay: float = 0.0,
        min_guidance_scale: float = 3.0,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        audio2audio_enable: bool = False,
        ref_audio_strength: float = 0.5,
        ref_audio_input: str = None,
        lora_name_or_path: str = "none",
        lora_weight: float = 1.0,
        retake_seeds: list = None,
        retake_variance: float = 0.5,
        task: str = "text2music",
        repaint_start: int = 0,
        repaint_end: int = 0,
        src_audio_path: str = None,
        edit_target_prompt: str = None,
        edit_target_lyrics: str = None,
        edit_n_min: float = 0.0,
        edit_n_max: float = 1.0,
        edit_n_avg: int = 1,
        save_path: str = None,
        batch_size: int = 1,
        debug: bool = False,
        extend_strength: float = 0.7,
        sample_rate: int = 48000,
        # ---- extend bootstrap controls ----
        extend_bootstrap: bool = True,
        extend_bootstrap_method: str = "a2a",  # "a2a" or "style"
        extend_bootstrap_strength: float = 0.6,
        seam_seconds: float = 0.75,
        extend_pad_mode: str = "boot_only",
        extend_tile_only: bool = False,   # <--- NEW
        extend_edge_bootstrap_only: bool = False,
        extend_tile_then_repaint: bool = False,
        # ---- extend bootstrap tuning (new) ----
        extend_bootstrap_edge_sec: float = 2.0,
        extend_bootstrap_sigma_max: Optional[float] = None,
        extend_bootstrap_noise_mode: str = "zeros",
    ):

        start_time = time.time()

        # Debug initialization
        if debug:
            # `debug` is your existing call arg; keep it to toggle
            self.debug.enabled = True
            self.debug.run_tag = f"{int(time.time())}"
            logger.info(f"[DBG] debug enabled tag={self.debug.run_tag}")

        if self.debug.enabled:
            logger.info(f"[PATH] task={task} a2a={audio2audio_enable} extend_tile_only={extend_tile_only}")

        # Sanity log to see effective task
        logger.info(f"EFFECTIVE TASK: {task}  (a2a_enable={audio2audio_enable}, ref_audio_input={'yes' if ref_audio_input else 'no'})")

        if not self.loaded:
            logger.warning("Checkpoint not loaded, loading checkpoint...")
            if self.quantized:
                self.load_quantized_checkpoint(self.checkpoint_dir)
            else:
                self.load_checkpoint(self.checkpoint_dir)
            
            # Environment logging
            cuda_ver = getattr(torch.version, "cuda", None)
            logger.info(f"[ENV] device={self.device} dtype={self.dtype} torch={torch.__version__} "
                        f"cuda={cuda_ver} mps={'yes' if torch.backends.mps.is_available() else 'no'}")

        self.load_lora(lora_name_or_path, lora_weight)
        load_model_cost = time.time() - start_time
        logger.info(f"Model loaded in {load_model_cost:.2f} seconds.")

        start_time = time.time()

        random_generators, actual_seeds = self.set_seeds(batch_size, manual_seeds)
        retake_random_generators, actual_retake_seeds = self.set_seeds(
            batch_size, retake_seeds
        )

        if isinstance(oss_steps, str) and len(oss_steps) > 0:
            oss_steps = list(map(int, oss_steps.split(",")))
        else:
            oss_steps = []

        # Validate required prompts
        if task != "edit" and not prompt:
            raise ValueError("`prompt` is required for task='text2music' / 'retake' / 'repaint' / 'extend'")
        if task == "edit" and not edit_target_prompt:
            raise ValueError("`edit_target_prompt` is required for task='edit'")
        
        # Handle None prompt to prevent tokenizer crash
        if prompt is None:
            prompt = ""
        texts = [prompt]
        t = _Tick("encode_text")
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(texts)
        _nan_guard("encoder_text_hidden_states", encoder_text_hidden_states)
        t.done()
        encoder_text_hidden_states = encoder_text_hidden_states.repeat(batch_size, 1, 1)
        text_attention_mask = text_attention_mask.repeat(batch_size, 1)

        encoder_text_hidden_states_null = None
        if use_erg_tag:
            t = _Tick("encode_text_null")
            encoder_text_hidden_states_null = self.get_text_embeddings_null(texts)
            t.done()
            encoder_text_hidden_states_null = encoder_text_hidden_states_null.repeat(batch_size, 1, 1)

        # not support for released checkpoint
        speaker_embeds = torch.zeros(batch_size, 512).to(self.device).to(self.dtype)

        # 6 lyric
        lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        lyric_mask = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        if lyrics:
            lyric_token_idx = self.tokenize_lyrics(lyrics, debug=debug)
            lyric_mask = [1] * len(lyric_token_idx)
            lyric_token_idx = (
                torch.tensor(lyric_token_idx)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )
            lyric_mask = (
                torch.tensor(lyric_mask)
                .unsqueeze(0)
                .to(self.device)
                .repeat(batch_size, 1)
            )

        if audio_duration <= 0:
            audio_duration = random.uniform(30.0, 240.0)
            logger.info(f"random audio duration: {audio_duration}")

        end_time = time.time()
        preprocess_time_cost = end_time - start_time
        start_time = end_time

        add_retake_noise = task in ("retake", "repaint", "extend")
        # retake equal to repaint
        if task == "retake":
            repaint_start = 0
            repaint_end = audio_duration

        src_latents = None
        if task in ("repaint", "edit", "extend"):
            assert src_audio_path is not None, "src_audio_path is required for retake/repaint/extend task"
            assert os.path.exists(
                src_audio_path
            ), f"src_audio_path {src_audio_path} does not exist"
            src_latents = self.infer_latents(src_audio_path)
        elif src_audio_path is not None:
            # Handle case where src_audio_path is provided but task doesn't require it
            src_latents = self.infer_latents(src_audio_path)
        
        ref_latents = None
        if ref_audio_input is not None:
            assert os.path.exists(
                ref_audio_input
            ), f"ref_audio_input {ref_audio_input} does not exist"
            ref_latents = self.infer_latents(ref_audio_input)

        if task == "edit":
            texts = [edit_target_prompt]
            target_encoder_text_hidden_states, target_text_attention_mask = (
                self.get_text_embeddings(texts)
            )
            target_encoder_text_hidden_states = (
                target_encoder_text_hidden_states.repeat(batch_size, 1, 1)
            )
            target_text_attention_mask = target_text_attention_mask.repeat(
                batch_size, 1
            )

            target_lyric_token_idx = (
                torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
            )
            target_lyric_mask = (
                torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
            )
            if edit_target_lyrics:
                target_lyric_token_idx = self.tokenize_lyrics(
                    edit_target_lyrics, debug=True
                )
                target_lyric_mask = [1] * len(target_lyric_token_idx)
                target_lyric_token_idx = (
                    torch.tensor(target_lyric_token_idx)
                    .unsqueeze(0)
                    .to(self.device)
                    .repeat(batch_size, 1)
                )
                target_lyric_mask = (
                    torch.tensor(target_lyric_mask)
                    .unsqueeze(0)
                    .to(self.device)
                    .repeat(batch_size, 1)
                )

            target_speaker_embeds = speaker_embeds.clone()

            target_latents = self.flowedit_diffusion_process(
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embds=speaker_embeds,
                lyric_token_ids=lyric_token_idx,
                lyric_mask=lyric_mask,
                target_encoder_text_hidden_states=target_encoder_text_hidden_states,
                target_text_attention_mask=target_text_attention_mask,
                target_speaker_embeds=target_speaker_embeds,
                target_lyric_token_ids=target_lyric_token_idx,
                target_lyric_mask=target_lyric_mask,
                src_latents=src_latents,
                random_generators=retake_random_generators,  # more diversity
                infer_steps=infer_step,
                guidance_scale=guidance_scale,
                n_min=edit_n_min,
                n_max=edit_n_max,
                n_avg=edit_n_avg,
                scheduler_type=scheduler_type,
            )
        else:
            t = _Tick("text2music_diffusion")
            target_latents = self.text2music_diffusion_process(
                duration=audio_duration,
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embds=speaker_embeds,
                lyric_token_ids=lyric_token_idx,
                lyric_mask=lyric_mask,
                guidance_scale=guidance_scale,
                omega_scale=omega_scale,
                infer_steps=infer_step,
                random_generators=random_generators,
                scheduler_type=scheduler_type,
                cfg_type=cfg_type,
                guidance_interval=guidance_interval,
                guidance_interval_decay=guidance_interval_decay,
                min_guidance_scale=min_guidance_scale,
                oss_steps=oss_steps,
                encoder_text_hidden_states_null=encoder_text_hidden_states_null,
                use_erg_lyric=use_erg_lyric,
                use_erg_diffusion=use_erg_diffusion,
                retake_random_generators=retake_random_generators,
                retake_variance=retake_variance,
                add_retake_noise=add_retake_noise,
                guidance_scale_text=guidance_scale_text,
                guidance_scale_lyric=guidance_scale_lyric,
                repaint_start=repaint_start,
                repaint_end=repaint_end,
                src_latents=src_latents,
                audio2audio_enable=audio2audio_enable,
                ref_audio_strength=ref_audio_strength,
                ref_latents=ref_latents,
                extend_strength=extend_strength,
                sample_rate=sample_rate,
                extend_bootstrap=extend_bootstrap,
                extend_bootstrap_method=extend_bootstrap_method,
                extend_bootstrap_strength=extend_bootstrap_strength,
                seam_seconds=seam_seconds,
                extend_pad_mode=extend_pad_mode,
                extend_tile_only=extend_tile_only,
                extend_edge_bootstrap_only=extend_edge_bootstrap_only,
                extend_tile_then_repaint=extend_tile_then_repaint,
                extend_bootstrap_edge_sec=extend_bootstrap_edge_sec,
                extend_bootstrap_sigma_max=extend_bootstrap_sigma_max,
                extend_bootstrap_noise_mode=extend_bootstrap_noise_mode,
            )
            t.done()

        end_time = time.time()
        diffusion_time_cost = end_time - start_time
        start_time = end_time

        t = _Tick("latents2audio")
        output_paths = self.latents2audio(
            latents=target_latents,
            target_wav_duration_second=audio_duration,  # use the value passed in
            save_path=save_path,
            format=format,
            sample_rate=sample_rate,
        )
        t.done()

        # Clean up memory after generation
        self.cleanup_memory()

        end_time = time.time()
        latent2audio_time_cost = end_time - start_time
        timecosts = {
            "preprocess": preprocess_time_cost,
            "diffusion": diffusion_time_cost,
            "latent2audio": latent2audio_time_cost,
        }

        input_params_json = {
            "format": format,
            "lora_name_or_path": lora_name_or_path,
            "lora_weight": lora_weight,
            "task": task,
            "prompt": prompt if task != "edit" else edit_target_prompt,
            "lyrics": lyrics if task != "edit" else edit_target_lyrics,
            "audio_duration": audio_duration,
            "infer_step": infer_step,
            "guidance_scale": guidance_scale,
            "scheduler_type": scheduler_type,
            "cfg_type": cfg_type,
            "omega_scale": omega_scale,
            "guidance_interval": guidance_interval,
            "guidance_interval_decay": guidance_interval_decay,
            "min_guidance_scale": min_guidance_scale,
            "use_erg_tag": use_erg_tag,
            "use_erg_lyric": use_erg_lyric,
            "use_erg_diffusion": use_erg_diffusion,
            "oss_steps": oss_steps,
            "timecosts": timecosts,
            "actual_seeds": actual_seeds,
            "retake_seeds": actual_retake_seeds,
            "retake_variance": retake_variance,
            "guidance_scale_text": guidance_scale_text,
            "guidance_scale_lyric": guidance_scale_lyric,
            "repaint_start": repaint_start,
            "repaint_end": repaint_end,
            "edit_n_min": edit_n_min,
            "edit_n_max": edit_n_max,
            "edit_n_avg": edit_n_avg,
            "src_audio_path": src_audio_path,
            "edit_target_prompt": edit_target_prompt,
            "edit_target_lyrics": edit_target_lyrics,
            "audio2audio_enable": audio2audio_enable,
            "ref_audio_strength": ref_audio_strength,
            "ref_audio_input": ref_audio_input,
        }
        # save input_params_json
        for output_audio_path in output_paths:
            input_params_json_save_path = output_audio_path.replace(
                f".{format}", "_input_params.json"
            )
            input_params_json["audio_path"] = output_audio_path
            with open(input_params_json_save_path, "w", encoding="utf-8") as f:
                json.dump(input_params_json, f, indent=4, ensure_ascii=False)

        return output_paths + [input_params_json]
