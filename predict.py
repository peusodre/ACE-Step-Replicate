"""
Replicate prediction interface for ACE-Step music generation model.
Supports full-length songs, lyric alignment, continuation, style transfer, voice cloning, and more.
"""

import os
import sys
import tempfile
import torch
from typing import Optional, List, Union
from cog import BasePredictor, Input, Path, File

# Add the current directory to the Python path so we can import acestep
sys.path.insert(0, '/src')

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

        # Initialize the ACE-Step pipeline to match Gradio implementation exactly
        self.pipeline = ACEStepPipeline(
            checkpoint_dir=None,  # Will download automatically
            dtype=dtype,
            persistent_storage_path="/tmp/acestep_cache",  # Match Gradio's persistent storage
            torch_compile=False,  # Match Gradio's torch_compile setting
        )
        
        # Load the model checkpoint
        self.pipeline.load_checkpoint()
        
    def predict(
        self,
        # === BASIC GENERATION ===
        task: str = Input(
            description="Generation task type",
            default="text2music",
            choices=["text2music", "audio2audio", "continuation", "inpainting", "style_transfer", "vocal_accompaniment"]
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
        input_audio: File = Input(
            description="Input audio file for audio2audio, continuation, inpainting, or style transfer (drag & drop supported)",
            default=None
        ),
        reference_audio: File = Input(
            description="Reference audio file for style transfer or voice cloning (drag & drop supported)",
            default=None
        ),
        
        # === CONTINUATION & INPAINTING ===
        continuation_mode: str = Input(
            description="How to continue/extend the input audio",
            default="extend_end",
            choices=["extend_start", "extend_end", "extend_both", "inpaint_middle"]
        ),
        inpaint_start_time: float = Input(
            description="Start time (seconds) for inpainting/editing section",
            default=0.0,
            ge=0.0
        ),
        inpaint_end_time: float = Input(
            description="End time (seconds) for inpainting/editing section",
            default=0.0,
            ge=0.0
        ),
        extend_duration: float = Input(
            description="Duration (seconds) to extend the audio",
            default=30.0,
            ge=5.0,
            le=120.0
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
            description="Random seed for reproducible generation (-1 for random)",
            default=-1
        ),
        variation_seed: int = Input(
            description="Seed for variation generation (-1 for random)",
            default=-1
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
        
        # Validate and process inputs based on task type
        self._validate_inputs(task, input_audio, reference_audio, inpaint_start_time, inpaint_end_time)
        
        # Set up generation parameters
        if task == "style_transfer":
            final_prompt = style_prompt if style_prompt else prompt
            final_lyrics = style_lyrics if style_lyrics else lyrics
        elif task in ["extend", "repaint"]:
            # For extend and repaint, use the original prompt or a generic one
            if task == "extend":
                final_prompt = prompt if prompt else "Continue the music in the same style and mood"
            else:  # repaint
                final_prompt = prompt if prompt else "Edit this section of the music"
            final_lyrics = lyrics if lyrics else ""
        else:
            final_prompt = prompt
            final_lyrics = lyrics
        
        # Set manual seeds (-1 means random)
        manual_seeds = [seed] if seed != -1 else None
        retake_seeds = [variation_seed] if variation_seed != -1 else manual_seeds
        
        try:
            # Save File objects to temporary files if provided
            input_audio_path = None
            reference_audio_path = None
            input_audio_duration = audio_duration  # Default to provided duration
            
            if input_audio:
                input_audio_path = os.path.join(temp_dir, "input_audio.wav")
                with open(input_audio_path, "wb") as f:
                    f.write(input_audio.read())
                
                # Get actual duration of input audio for extend/repaint tasks
                if task in ["extend", "repaint"]:
                    import librosa
                    input_audio_duration = librosa.get_duration(path=input_audio_path)
                    print(f"DEBUG: Input audio duration: {input_audio_duration} seconds")
            
            if reference_audio:
                reference_audio_path = os.path.join(temp_dir, "reference_audio.wav")
                with open(reference_audio_path, "wb") as f:
                    f.write(reference_audio.read())
            
            # Configure task-specific parameters
            task_params = self._configure_task_parameters(
                task=task,
                input_audio=input_audio_path,
                reference_audio=reference_audio_path,
                inpaint_start_time=inpaint_start_time,
                inpaint_end_time=inpaint_end_time,
                extend_duration=extend_duration,
                style_strength=style_strength,
                audio2audio_strength=audio2audio_strength,
                variation_strength=variation_strength,
                generate_accompaniment=generate_accompaniment,
                accompaniment_style=accompaniment_style,
                audio_duration=input_audio_duration,  # Use actual input audio duration
                continuation_mode=continuation_mode
            )
            
            # Debug logging for extend and repaint
            if task in ["extend", "repaint"]:
                print(f"DEBUG: Task: {task}")
                print(f"DEBUG: Task type: {task_params['task_type']}")
                print(f"DEBUG: Source audio: {task_params.get('src_audio_path')}")
                print(f"DEBUG: Repaint start: {task_params.get('repaint_start')}")
                print(f"DEBUG: Repaint end: {task_params.get('repaint_end')}")
                print(f"DEBUG: Audio duration: {task_params['audio_duration']}")
                print(f"DEBUG: Prompt: {final_prompt}")
                print(f"DEBUG: Lyrics: {final_lyrics}")
                print(f"DEBUG: Continuation mode: {continuation_mode}")
                
                # Check if source audio file exists
                if task_params.get('src_audio_path'):
                    if os.path.exists(task_params['src_audio_path']):
                        print(f"DEBUG: Source audio file exists and is readable")
                    else:
                        print(f"DEBUG: ERROR - Source audio file does not exist!")
                else:
                    print(f"DEBUG: ERROR - No source audio path provided!")
                
                # Additional debug for repaint parameters
                if task == "repaint":
                    print(f"DEBUG: Repaint range: {task_params.get('repaint_start')} to {task_params.get('repaint_end')} seconds")
                    print(f"DEBUG: Total audio duration: {task_params['audio_duration']} seconds")
                    print(f"DEBUG: Repaint percentage: {task_params.get('repaint_start')/task_params['audio_duration']*100:.1f}% to {task_params.get('repaint_end')/task_params['audio_duration']*100:.1f}%")
                    print(f"DEBUG: Inpaint start time: {inpaint_start_time}")
                    print(f"DEBUG: Inpaint end time: {inpaint_end_time}")
            
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
                src_audio_path=task_params.get("src_audio_path"),
                ref_audio_input=task_params.get("ref_audio_input"),
                audio2audio_enable=task_params.get("audio2audio_enable", False),
                ref_audio_strength=task_params.get("ref_audio_strength", 0.5),
                repaint_start=int(task_params.get("repaint_start", 0)),
                repaint_end=int(task_params.get("repaint_end", 0)),
                
                
                # Variation and retake parameters
                retake_seeds=retake_seeds,
                retake_variance=1.0 if task == "extend" else (0.2 if task == "repaint" else variation_strength),
                
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
                oss_steps="",
            )
            
            # Return the first audio file (the model returns [audio_path, params_json])
            audio_path = output_paths[0]
            return Path(audio_path)
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed for task '{task}': {str(e)}")
    
    def _validate_inputs(self, task: str, input_audio: str, reference_audio: str, 
                       inpaint_start_time: float, inpaint_end_time: float) -> None:
        """Validate inputs based on the selected task."""
        
        if task in ["audio2audio", "extend", "repaint", "continuation", "inpainting", "style_transfer"] and not input_audio:
            raise ValueError(f"Task '{task}' requires input_audio to be provided")
        
        if task in ["repaint", "inpainting"] and inpaint_end_time <= inpaint_start_time:
            raise ValueError("inpaint_end_time must be greater than inpaint_start_time")
        
        if task == "vocal_accompaniment" and not input_audio:
            raise ValueError("Task 'vocal_accompaniment' requires input_audio with vocals")
    
    def _configure_task_parameters(self, task: str, input_audio: str, reference_audio: str,
                                 inpaint_start_time: float, inpaint_end_time: float,
                                 extend_duration: float, style_strength: float, audio2audio_strength: float,
                                 variation_strength: float, generate_accompaniment: bool, 
                                 accompaniment_style: str, audio_duration: float, continuation_mode: str) -> dict:
        """Configure task-specific parameters for the pipeline."""
        
        params = {
            "audio_duration": audio_duration,
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
            # Audio extension with different modes
            total_duration = audio_duration + extend_duration
            transition_overlap = min(3.0, audio_duration * 0.1)  # 3 seconds or 10% of original audio, whichever is smaller
            
            if continuation_mode == "extend_start":
                # Extend from the beginning - repaint the first part
                repaint_start = 0
                repaint_end = int(extend_duration + transition_overlap)
                
            elif continuation_mode == "extend_end":
                # Extend from the end - repaint the last part with smooth transition
                repaint_start = max(0, int(audio_duration - transition_overlap))
                repaint_end = int(total_duration)
                
            elif continuation_mode == "extend_both":
                # Extend from both sides - repaint both ends
                repaint_start = 0
                repaint_end = int(total_duration)
                
            elif continuation_mode == "inpaint_middle":
                # Inpaint the middle section instead of extending
                middle_start = int(audio_duration * 0.25)  # Start at 25% of original audio
                middle_end = int(audio_duration * 0.75)   # End at 75% of original audio
                repaint_start = middle_start
                repaint_end = middle_end
                total_duration = audio_duration  # Keep original duration for inpainting
                
            else:
                # Default to extend_end
                repaint_start = max(0, int(audio_duration - transition_overlap))
                repaint_end = int(total_duration)
            
            params.update({
                "task_type": "extend",
                "src_audio_path": input_audio,
                "audio_duration": total_duration,
                "repaint_start": repaint_start,
                "repaint_end": repaint_end,
            })
        
        elif task == "repaint":
            # Audio repainting - match Gradio implementation exactly
            # Ensure repaint parameters are within valid range
            repaint_start = max(0, int(inpaint_start_time))
            repaint_end = min(int(audio_duration), int(inpaint_end_time))
            
            # Ensure repaint_end > repaint_start
            if repaint_end <= repaint_start:
                repaint_end = repaint_start + 1
            
            params.update({
                "task_type": "repaint",
                "src_audio_path": input_audio,
                "audio_duration": audio_duration,  # Use actual input audio duration
                "repaint_start": repaint_start,
                "repaint_end": repaint_end,
            })
        
        elif task == "style_transfer":
            # Style transfer using edit mode
            params.update({
                "task_type": "edit",
                "src_audio_path": input_audio,
                "edit_target_prompt": prompt,  # Use the provided prompt as target
                "edit_target_lyrics": lyrics,  # Use the provided lyrics as target
                "edit_n_min": 1.0 - style_strength,  # Higher strength = lower n_min
                "edit_n_max": 1.0,
                "edit_n_avg": 1,
            })
            
            if reference_audio:
                # Use reference audio for style guidance
                params["ref_audio_input"] = reference_audio
                params["ref_audio_strength"] = 0.3  # Moderate influence
        
        elif task == "vocal_accompaniment":
            # Generate accompaniment for vocals
            # This would require specialized LoRA or controlnet in full implementation
            params.update({
                "task_type": "text2music",
                "src_audio_path": input_audio,
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
        
        return params
lly 