"""
Replicate prediction interface for ACE-Step music generation model.
Supports full-length songs, lyric alignment, continuation, style transfer, voice cloning, and more.
"""

import os
import tempfile
import torch
from typing import Optional, List
from cog import BasePredictor, Input, Path

from acestep.pipeline_ace_step import ACEStepPipeline


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Set environment for CPU offloading to manage memory efficiently
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Choose safest dtype based on hardware support
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

        # Initialize the ACE-Step pipeline with optimizations for inference
        self.pipeline = ACEStepPipeline(
            checkpoint_dir=None,  # Will download automatically
            device_id=0,
            dtype=dtype,
            torch_compile=True,  # Optimize for faster inference
            cpu_offload=True,    # Enable CPU offloading to manage GPU memory
            overlapped_decode=True,  # Enable overlapped decoding for efficiency
        )
        
        # Load the model checkpoint
        self.pipeline.load_checkpoint()
        
    def predict(
        self,
        # === BASIC GENERATION ===
        task: str = Input(
            description="Generation task type",
            default="text2music",
            choices=["text2music", "audio2audio", "extend", "repaint", "continuation", "inpainting", "style_transfer", "vocal_accompaniment"]
        ),
        prompt: str = Input(
            description="Text prompt describing the music style, genre, or mood",
            default="Electronic dance music, upbeat, energetic"
        ),
        lyrics: str = Input(
            description="Lyrics for the song (optional). Use structure tags like [verse], [chorus], [bridge]",
            default=""
        ),
        audio_duration: float = Input(
            description="Duration of the generated audio in seconds (up to 4 minutes for full songs)",
            default=30.0,
            ge=10.0,
            le=240.0
        ),
        
        # === AUDIO INPUT FOR ADVANCED TASKS ===
        input_audio: Path = Input(
            description="Input audio file for audio2audio, continuation, inpainting, or style transfer",
            default=None
        ),
        reference_audio: Path = Input(
            description="Reference audio for style transfer or voice cloning",
            default=None
        ),
        
        # === EXTENSION & REPAINTING ===
        inpaint_start_time: float = Input(
            description="Start time (seconds) for repainting/editing section",
            default=0.0,
            ge=0.0
        ),
        inpaint_end_time: float = Input(
            description="End time (seconds) for repainting/editing section",
            default=0.0,
            ge=0.0
        ),
        repaint_strength: float = Input(
            description="Strength of repaint changes (0.0 = no change, 1.0 = maximum change)",
            default=0.5,
            ge=0.0,
            le=1.0
        ),
        extend_duration: float = Input(
            description="Duration (seconds) to extend the audio (extends both left and right)",
            default=30.0,
            ge=5.0,
            le=120.0
        ),
        extend_strength: float = Input(
            description="Strength of input audio influence for continuation (0.0 = ignore input, 1.0 = closely follow input style)",
            default=0.7,
            ge=0.0,
            le=1.0
        ),
        
        # === STYLE TRANSFER & REMIX ===
        style_prompt: str = Input(
            description="Target style prompt for style transfer/remix (if different from main prompt)",
            default=""
        ),
        style_lyrics: str = Input(
            description="Target lyrics for style transfer/remix (if different from main lyrics)",
            default=""
        ),
        style_strength: float = Input(
            description="Strength of style transfer (0.0 = keep original, 1.0 = full transfer)",
            default=0.7,
            ge=0.0,
            le=1.0
        ),
        
        # === AUDIO2AUDIO & VOICE CLONING ===
        audio2audio_strength: float = Input(
            description="Strength of audio2audio transformation (0.0 = no change, 1.0 = full generation)",
            default=0.5,
            ge=0.0,
            le=1.0
        ),
        preserve_vocals: bool = Input(
            description="Try to preserve vocal characteristics during transformation",
            default=False
        ),
        
        # === STEM GENERATION & ACCOMPANIMENT ===
        generate_accompaniment: bool = Input(
            description="Generate accompaniment for vocals (requires input_audio with vocals)",
            default=False
        ),
        accompaniment_style: str = Input(
            description="Style for generated accompaniment",
            default="matching",
            choices=["matching", "jazz", "rock", "electronic", "classical", "pop", "folk"]
        ),
        
        # === ADVANCED GENERATION PARAMETERS ===
        infer_steps: int = Input(
            description="Number of inference steps (higher = better quality, slower)",
            default=60,
            ge=20,
            le=100
        ),
        guidance_scale: float = Input(
            description="Guidance scale for conditioning (higher = more adherence to prompt)",
            default=15.0,
            ge=1.0,
            le=30.0
        ),
        scheduler_type: str = Input(
            description="Scheduler type for the diffusion process",
            default="euler",
            choices=["euler", "heun", "pingpong"]
        ),
        cfg_type: str = Input(
            description="Classifier-free guidance type",
            default="apg",
            choices=["apg", "cfg", "cfg_star"]
        ),
        omega_scale: float = Input(
            description="Omega scale parameter for the scheduler",
            default=10.0,
            ge=1.0,
            le=20.0
        ),
        
        # === REPRODUCIBILITY & VARIATION ===
        seed: int = Input(
            description="Random seed for reproducible generation (optional)",
            default=None
        ),
        variation_seed: int = Input(
            description="Seed for variation generation (creates slight variations)",
            default=None
        ),
        variation_strength: float = Input(
            description="Strength of variation (0.0 = no variation, 1.0 = maximum variation)",
            default=0.0,
            ge=0.0,
            le=1.0
        ),
        
        # === ENHANCED GUIDANCE CONTROLS ===
        use_erg_tag: bool = Input(
            description="Use ERG (Enhanced Representation Guidance) for tags",
            default=True
        ),
        use_erg_lyric: bool = Input(
            description="Use ERG for lyrics",
            default=True
        ),
        use_erg_diffusion: bool = Input(
            description="Use ERG during diffusion",
            default=True
        ),
        guidance_scale_text: float = Input(
            description="Separate guidance scale for text (0.0 = use main guidance_scale)",
            default=0.0,
            ge=0.0,
            le=30.0
        ),
        guidance_scale_lyric: float = Input(
            description="Separate guidance scale for lyrics (0.0 = use main guidance_scale)",
            default=0.0,
            ge=0.0,
            le=30.0
        ),
        
        # === LORA & MODEL CUSTOMIZATION ===
        lora_name_or_path: str = Input(
            description="LoRA adapter name or path (use 'none' for base model, 'ACE-Step/ACE-Step-v1-chinese-rap-LoRA' for rap)",
            default="none"
        ),
        lora_weight: float = Input(
            description="LoRA adapter weight",
            default=1.0,
            ge=0.0,
            le=2.0
        ),
        
        # === OUTPUT SETTINGS ===
        output_format: str = Input(
            description="Output audio format",
            default="wav",
            choices=["wav", "mp3", "ogg"]
        ),
        sample_rate: int = Input(
            description="Output sample rate (Hz)",
            default=48000,
            choices=[44100, 48000]
        )
    ) -> Path:
        """Run a single prediction on the model with advanced features"""
        
        # Create a temporary directory for output
        temp_dir = tempfile.mkdtemp()
        
        # Map old task names to new ones for backward compatibility
        if task == "continuation":
            task = "extend"
        elif task == "inpainting":
            task = "repaint"
        
        # Save File objects to temporary files if provided
        input_audio_path = None
        reference_audio_path = None
        
        if input_audio is not None:
            input_audio_path = os.path.join(temp_dir, "input_audio.wav")
            with open(input_audio_path, "wb") as f:
                with open(input_audio, "rb") as input_file:
                    f.write(input_file.read())
        
        if reference_audio is not None:
            reference_audio_path = os.path.join(temp_dir, "reference_audio.wav")
            with open(reference_audio_path, "wb") as f:
                with open(reference_audio, "rb") as ref_file:
                    f.write(ref_file.read())
        
        # Validate and process inputs based on task type
        self._validate_inputs(task, input_audio_path, reference_audio_path, inpaint_start_time, inpaint_end_time)
        
        # Debug: Print values to understand what's happening
        print(f"DEBUG: Task: {task}")
        print(f"DEBUG: input_audio: {input_audio}")
        print(f"DEBUG: input_audio_path: {input_audio_path}")
        
        # Ensure input_audio_path is set for tasks that require it
        if task in ["extend", "repaint", "style_transfer", "vocal_accompaniment"] and input_audio_path is None:
            raise ValueError(f"Task '{task}' requires input_audio to be provided")
        
        # Set up generation parameters
        final_prompt = style_prompt if style_prompt and task == "style_transfer" else prompt
        final_lyrics = style_lyrics if style_lyrics and task == "style_transfer" else lyrics
        
        # Set manual seeds
        manual_seeds = [seed] if seed is not None else None
        retake_seeds = [variation_seed] if variation_seed is not None else manual_seeds
        
        try:
            # Configure task-specific parameters
            task_params = self._configure_task_parameters(
                task=task,
                input_audio=input_audio,
                reference_audio=reference_audio,
                inpaint_start_time=inpaint_start_time,
                inpaint_end_time=inpaint_end_time,
                repaint_strength=repaint_strength,
                extend_duration=extend_duration,
                extend_strength=extend_strength,
                style_strength=style_strength,
                audio2audio_strength=audio2audio_strength,
                variation_strength=variation_strength,
                generate_accompaniment=generate_accompaniment,
                accompaniment_style=accompaniment_style,
                audio_duration=audio_duration,
                input_audio_path=input_audio_path,
                reference_audio_path=reference_audio_path
            )
            
            # Debug: Print task_params to see what's in it
            print(f"DEBUG: task_params keys: {list(task_params.keys())}")
            print(f"DEBUG: task_params['src_audio_path']: {task_params.get('src_audio_path')}")
            print(f"DEBUG: task_params.get('src_audio_path'): {task_params.get('src_audio_path')}")
            
            # Debug: Print the actual src_audio_path value being passed to pipeline
            src_audio_path_value = task_params.get("src_audio_path")
            print(f"DEBUG: About to call pipeline with src_audio_path: {src_audio_path_value}")
            print(f"DEBUG: Type of src_audio_path: {type(src_audio_path_value)}")
            
            # Run the ACE-Step pipeline with all parameters
            output_paths = self.pipeline(
                # Basic parameters
                format=output_format,
                audio_duration=task_params["audio_duration"],
                prompt=final_prompt,
                lyrics=final_lyrics,
                infer_step=infer_steps,
                guidance_scale=guidance_scale,
                scheduler_type=scheduler_type,
                cfg_type=cfg_type,
                omega_scale=omega_scale,
                manual_seeds=manual_seeds,
                
                # Advanced guidance parameters
                guidance_interval=0.5,
                guidance_interval_decay=0.0,
                min_guidance_scale=3.0,
                use_erg_tag=use_erg_tag,
                use_erg_lyric=use_erg_lyric,
                use_erg_diffusion=use_erg_diffusion,
                guidance_scale_text=guidance_scale_text,
                guidance_scale_lyric=guidance_scale_lyric,
                
                # Task-specific parameters
                task=task_params["task_type"],
                src_audio_path=src_audio_path_value,  # Use the debug variable
                ref_audio_input=task_params.get("ref_audio_input"),
                audio2audio_enable=task_params.get("audio2audio_enable", False),
                ref_audio_strength=task_params.get("ref_audio_strength", 0.5),
                repaint_start=task_params.get("repaint_start", 0),
                repaint_end=task_params.get("repaint_end", 0),
                
                # Variation and retake parameters
                retake_seeds=retake_seeds,
                retake_variance=task_params.get("retake_variance", variation_strength),
                
                # Style transfer parameters (for edit mode)
                edit_target_prompt=task_params.get("edit_target_prompt"),
                edit_target_lyrics=task_params.get("edit_target_lyrics"),
                edit_n_min=task_params.get("edit_n_min", 0.0),
                edit_n_max=task_params.get("edit_n_max", 1.0),
                edit_n_avg=task_params.get("edit_n_avg", 1),
                
                # Model customization
                lora_name_or_path=lora_name_or_path,
                lora_weight=lora_weight,
                
                # Output settings
                save_path=temp_dir,
                batch_size=1,
                debug=False,
                
                # Additional parameters
                oss_steps=[],
            )
            
            # Return the first audio file (the model returns [audio_path, params_json])
            audio_path = output_paths[0]
            return Path(audio_path)
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed for task '{task}': {str(e)}")
    
    def _validate_inputs(self, task: str, input_audio: Optional[Path], reference_audio: Optional[Path], 
                        inpaint_start_time: float, inpaint_end_time: float) -> None:
        """Validate inputs based on the selected task."""
        
        if task in ["audio2audio", "extend", "repaint", "continuation", "inpainting", "style_transfer"] and input_audio is None:
            raise ValueError(f"Task '{task}' requires input_audio to be provided")
        
        if task in ["repaint", "inpainting"] and inpaint_end_time <= inpaint_start_time:
            raise ValueError("inpaint_end_time must be greater than inpaint_start_time")
        
        if task == "vocal_accompaniment" and input_audio is None:
            raise ValueError("Task 'vocal_accompaniment' requires input_audio with vocals")
    
    def _configure_task_parameters(self, task: str, input_audio: Optional[Path], reference_audio: Optional[Path],
                                 inpaint_start_time: float, inpaint_end_time: float, repaint_strength: float,
                                 extend_duration: float, extend_strength: float, style_strength: float, 
                                 audio2audio_strength: float, variation_strength: float, generate_accompaniment: bool, 
                                 accompaniment_style: str, audio_duration: float, input_audio_path: Optional[str] = None,
                                 reference_audio_path: Optional[str] = None) -> dict:
        """Configure task-specific parameters for the pipeline."""
        
        # Get actual audio duration for tasks that use input audio
        actual_audio_duration = audio_duration  # Default to provided duration
        if input_audio and task in ["extend", "repaint", "style_transfer", "vocal_accompaniment"]:
            import librosa
            # Convert Path to string for librosa
            audio_file_path = str(input_audio) if hasattr(input_audio, '__fspath__') else input_audio
            actual_audio_duration = librosa.get_duration(path=audio_file_path)
        
        params = {
            "audio_duration": actual_audio_duration,
            "task_type": "text2music",  # Default ACE-Step task type
        }
        
        if task == "text2music":
            # Standard text-to-music generation
            if variation_strength > 0:
                params["task_type"] = "retake"
        
        elif task == "audio2audio":
            # Audio-to-audio transformation
            params.update({
                "task_type": "text2music",
                "audio2audio_enable": True,
                "ref_audio_input": input_audio,
                "ref_audio_strength": 1.0 - audio2audio_strength,  # ACE-Step uses inverse strength
            })
        
        elif task == "extend":
            # Audio extension - match Gradio implementation exactly
            print(f"DEBUG: extend task - input_audio_path: {input_audio_path}")
            params.update({
                "task_type": "extend",
                "src_audio_path": input_audio_path,  # Use the temporary file path, not the Path object
                "repaint_start": -extend_duration,  # left_extend_length
                "repaint_end": actual_audio_duration + extend_duration,  # actual_audio_duration + right_extend_length
                "audio_duration": actual_audio_duration + 2 * extend_duration,  # Total output duration
                # Enable audio2audio to condition generation on input audio style
                "audio2audio_enable": True,
                "ref_audio_input": input_audio_path,  # Use the temporary file path
                "ref_audio_strength": extend_strength,  # User-controlled influence from input audio
            })
            print(f"DEBUG: extend task - params['src_audio_path']: {params.get('src_audio_path')}")
        
        elif task == "repaint":
            # Audio repainting - match Gradio implementation exactly
            params.update({
                "task_type": "repaint",
                "src_audio_path": input_audio_path,  # Use the temporary file path, not the Path object
                "repaint_start": inpaint_start_time,
                "repaint_end": inpaint_end_time,
                "audio_duration": actual_audio_duration,  # Use actual audio duration
            })
            # Set retake_variance based on repaint_strength for repaint tasks
            # This overrides the variation_strength parameter for repaint
            params["retake_variance"] = repaint_strength
        
        elif task == "style_transfer":
            # Style transfer using edit mode
            params.update({
                "task_type": "edit",
                "src_audio_path": input_audio_path,  # Use the temporary file path, not the Path object
                "edit_n_min": 1.0 - style_strength,  # Higher strength = lower n_min
                "edit_n_max": 1.0,
                "edit_n_avg": 1,
                "audio_duration": actual_audio_duration,  # Use actual audio duration
            })
            
            if reference_audio:
                # Use reference audio for style guidance
                params["ref_audio_input"] = reference_audio_path  # Use the temporary file path
                params["ref_audio_strength"] = 0.3  # Moderate influence
        
        elif task == "vocal_accompaniment":
            # Generate accompaniment for vocals
            # This would require specialized LoRA or controlnet in full implementation
            params.update({
                "task_type": "text2music",
                "src_audio_path": input_audio_path,  # Use the temporary file path, not the Path object
                "audio_duration": actual_audio_duration,  # Use actual audio duration
            })
            
            # Adjust prompt for accompaniment generation
            style_mapping = {
                "jazz": "Jazz accompaniment, swing rhythm, piano and bass",
                "rock": "Rock accompaniment, electric guitar and drums",
                "electronic": "Electronic accompaniment, synthesizers and beats",
                "classical": "Classical accompaniment, orchestra and strings",
                "pop": "Pop accompaniment, modern production",
                "folk": "Folk accompaniment, acoustic guitar and simple rhythm"
            }
            
            if accompaniment_style != "matching":
                # Override with specific accompaniment style
                params["edit_target_prompt"] = style_mapping.get(accompaniment_style, 
                                                               f"{accompaniment_style} accompaniment")
        
        # Debug: Print final params before returning
        print(f"DEBUG: Final params keys: {list(params.keys())}")
        print(f"DEBUG: Final params['src_audio_path']: {params.get('src_audio_path')}")
        
        return params
