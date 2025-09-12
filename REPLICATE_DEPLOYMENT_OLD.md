# ACE-Step Replicate Deployment Guide

This guide explains how to deploy the ACE-Step music generation model on Replicate.

## Overview

ACE-Step is a state-of-the-art music generation foundation model that can create high-quality music from text prompts and lyrics. This deployment makes it accessible through Replicate's platform for scalable inference.

## Features

- **🎵 Full-Length Songs**: Generate up to 4 minutes of coherent music in one pass
- **🎤 Lyric Alignment**: Explicit lyric conditioning for perfect word-music synchronization
- **🔄 Audio Continuation**: Extend existing songs at the beginning, end, or both sides
- **🎨 Audio Inpainting**: Fill in missing sections or edit specific parts of a song
- **🎭 Style Transfer**: Transform existing music to different styles while preserving structure
- **🎪 Voice Cloning**: Apply different vocal timbres and characteristics
- **🎹 Stem Generation**: Generate accompaniment for vocals or create individual instrument stems
- **⚡ Ultra-Fast**: ~20 seconds for 4 minutes of music on A100 GPU
- **🎯 High Fidelity**: Hybrid diffusion + transformer + compression for cleaner output
- **🌍 Multilingual**: Supports 19 languages including English, Chinese, Spanish, Japanese, etc.
- **🎼 Multiple Genres**: All mainstream music styles with various instrumentation
- **🔧 Advanced Controls**: ERG guidance, LoRA adapters, and fine-grained parameter control

## Prerequisites

1. **Replicate Account**: Sign up at [replicate.com](https://replicate.com)
2. **Cog Installed**: Install the Cog CLI tool
3. **Docker**: Ensure Docker is installed and running

## Installation

1. **Install Cog**:

   ```bash
   sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
   sudo chmod +x /usr/local/bin/cog
   ```

2. **Clone this repository** (if not already done):
   ```bash
   git clone https://github.com/ace-step/ACE-Step.git
   cd ACE-Step
   ```

## Deployment Steps

1. **Test locally** (optional but recommended):

   ```bash
   cog predict -i prompt="Electronic dance music, upbeat, energetic" -i audio_duration=30
   ```

2. **Login to Replicate**:

   ```bash
   cog login
   ```

3. **Create a model on Replicate**:

   - Go to [replicate.com/create](https://replicate.com/create)
   - Create a new model with your desired name

4. **Push the model**:
   ```bash
   cog push r8.im/your-username/ace-step
   ```
   Replace `your-username` with your Replicate username.

## Usage Examples

### 🎵 Basic Text-to-Music Generation

```python
import replicate

# Generate a 30-second electronic track
output = replicate.run(
    "your-username/ace-step",
    input={
        "task": "text2music",
        "prompt": "Electronic dance music, upbeat, energetic",
        "audio_duration": 30,
        "infer_steps": 60,
        "guidance_scale": 15.0
    }
)
```

### 🎤 Full Song with Lyrics (4 minutes)

```python
# Generate a complete song with structure and lyrics
output = replicate.run(
    "your-username/ace-step",
    input={
        "task": "text2music",
        "prompt": "Pop ballad, emotional, piano-driven, modern production",
        "lyrics": """[verse]
In the quiet of the night
I find my way to you
Through the shadows and the light
[chorus]
This is our moment
This is our time
Everything feels so right
[verse]
Stars are dancing in your eyes
Dreams are calling out our names
[chorus]
This is our moment
This is our time
Everything feels so right
[bridge]
When the world fades away
Only love will remain
[outro]
This is our time""",
        "audio_duration": 240,  # 4 minutes
        "infer_steps": 80,
        "guidance_scale": 18.0,
        "scheduler_type": "pingpong",  # Best for long coherent generation
        "seed": 42
    }
)
```

### 🔄 Audio Continuation / Extension

```python
# Extend an existing song at the end
output = replicate.run(
    "your-username/ace-step",
    input={
        "task": "continuation",
        "input_audio": open("my_song.wav", "rb"),
        "continuation_mode": "extend_end",
        "extend_duration": 60,
        "prompt": "Continue with same style, add guitar solo",
        "infer_steps": 60
    }
)
```

### 🎨 Audio Inpainting / Editing

```python
# Edit a specific section of a song
output = replicate.run(
    "your-username/ace-step",
    input={
        "task": "inpainting",
        "input_audio": open("original_song.wav", "rb"),
        "inpaint_start_time": 30.0,  # Start editing at 30 seconds
        "inpaint_end_time": 60.0,    # End editing at 60 seconds
        "prompt": "Add dramatic orchestral strings",
        "lyrics": "[bridge]\nEverything changes now\nNothing will be the same",
        "infer_steps": 70
    }
)
```

### 🎭 Style Transfer / Remix

```python
# Transform a song to a different style
output = replicate.run(
    "your-username/ace-step",
    input={
        "task": "style_transfer",
        "input_audio": open("pop_song.wav", "rb"),
        "style_prompt": "Jazz arrangement, swing rhythm, saxophone lead",
        "style_strength": 0.8,  # High transformation
        "preserve_vocals": True,
        "infer_steps": 75
    }
)
```

### 🎪 Voice Cloning / Audio2Audio

```python
# Apply voice characteristics from reference audio
output = replicate.run(
    "your-username/ace-step",
    input={
        "task": "audio2audio",
        "input_audio": open("my_vocals.wav", "rb"),
        "reference_audio": open("target_voice.wav", "rb"),
        "audio2audio_strength": 0.6,
        "prompt": "Maintain emotional expression, smooth vocal style",
        "preserve_vocals": True
    }
)
```

### 🎹 Vocal Accompaniment Generation

```python
# Generate instrumental backing for vocals
output = replicate.run(
    "your-username/ace-step",
    input={
        "task": "vocal_accompaniment",
        "input_audio": open("acapella_vocals.wav", "rb"),
        "accompaniment_style": "jazz",
        "prompt": "Rich jazz accompaniment with piano, bass, and light drums",
        "infer_steps": 70
    }
)
```

### 🎤 Rap Generation with LoRA

```python
# Use specialized rap LoRA adapter
output = replicate.run(
    "your-username/ace-step",
    input={
        "task": "text2music",
        "prompt": "Hip-hop beat, hard hitting bass, trap style",
        "lyrics": """[verse]
Started from the bottom now we here
Every single day we persevere
[chorus]
This is how we rise up
Never gonna give up
[verse]
Dreams become reality
Nothing can stop our energy""",
        "lora_name_or_path": "ACE-Step/ACE-Step-v1-chinese-rap-LoRA",
        "lora_weight": 1.0,
        "audio_duration": 120,
        "infer_steps": 60
    }
)
```

### ⚡ Quick Variations

```python
# Generate variations of existing music
output = replicate.run(
    "your-username/ace-step",
    input={
        "task": "text2music",
        "prompt": "Ambient electronic, atmospheric",
        "seed": 12345,
        "variation_seed": 67890,
        "variation_strength": 0.3,  # Subtle variation
        "audio_duration": 60
    }
)
```

## Input Parameters

### Core Parameters

| Parameter        | Type   | Default      | Description                                                                                       |
| ---------------- | ------ | ------------ | ------------------------------------------------------------------------------------------------- |
| `task`           | string | "text2music" | Task type: text2music, audio2audio, continuation, inpainting, style_transfer, vocal_accompaniment |
| `prompt`         | string | -            | Text describing music style, genre, or mood                                                       |
| `lyrics`         | string | ""           | Song lyrics with optional structure tags                                                          |
| `audio_duration` | float  | 30.0         | Duration in seconds (10-240, up to 4 minutes)                                                     |

### Audio Input Parameters

| Parameter         | Type | Default | Description                                      |
| ----------------- | ---- | ------- | ------------------------------------------------ |
| `input_audio`     | file | null    | Input audio for advanced tasks                   |
| `reference_audio` | file | null    | Reference audio for style transfer/voice cloning |

### Continuation & Inpainting

| Parameter            | Type   | Default      | Description                                                          |
| -------------------- | ------ | ------------ | -------------------------------------------------------------------- |
| `continuation_mode`  | string | "extend_end" | How to extend: extend_start, extend_end, extend_both, inpaint_middle |
| `inpaint_start_time` | float  | 0.0          | Start time (seconds) for inpainting section                          |
| `inpaint_end_time`   | float  | 0.0          | End time (seconds) for inpainting section                            |
| `extend_duration`    | float  | 30.0         | Duration (seconds) to extend                                         |

### Style Transfer & Remix

| Parameter        | Type   | Default | Description                            |
| ---------------- | ------ | ------- | -------------------------------------- |
| `style_prompt`   | string | ""      | Target style prompt for transformation |
| `style_lyrics`   | string | ""      | Target lyrics for transformation       |
| `style_strength` | float  | 0.7     | Strength of style transfer (0.0-1.0)   |

### Audio2Audio & Voice Cloning

| Parameter              | Type  | Default | Description                           |
| ---------------------- | ----- | ------- | ------------------------------------- |
| `audio2audio_strength` | float | 0.5     | Transformation strength (0.0-1.0)     |
| `preserve_vocals`      | bool  | false   | Try to preserve vocal characteristics |

### Accompaniment Generation

| Parameter                | Type   | Default    | Description                                                   |
| ------------------------ | ------ | ---------- | ------------------------------------------------------------- |
| `generate_accompaniment` | bool   | false      | Generate accompaniment for vocals                             |
| `accompaniment_style`    | string | "matching" | Style: matching, jazz, rock, electronic, classical, pop, folk |

### Generation Quality

| Parameter        | Type   | Default | Description                                       |
| ---------------- | ------ | ------- | ------------------------------------------------- |
| `infer_steps`    | int    | 60      | Inference steps (20-100, higher = better quality) |
| `guidance_scale` | float  | 15.0    | Conditioning strength (1.0-30.0)                  |
| `scheduler_type` | string | "euler" | Scheduler: "euler", "heun", or "pingpong"         |
| `cfg_type`       | string | "apg"   | Guidance type: "apg", "cfg", or "cfg_star"        |
| `omega_scale`    | float  | 10.0    | Omega parameter (1.0-20.0)                        |

### Reproducibility & Variation

| Parameter            | Type  | Default | Description                     |
| -------------------- | ----- | ------- | ------------------------------- |
| `seed`               | int   | null    | Random seed for reproducibility |
| `variation_seed`     | int   | null    | Seed for variation generation   |
| `variation_strength` | float | 0.0     | Variation strength (0.0-1.0)    |

### Enhanced Guidance

| Parameter              | Type  | Default | Description                                       |
| ---------------------- | ----- | ------- | ------------------------------------------------- |
| `use_erg_tag`          | bool  | true    | Enhanced representation guidance for tags         |
| `use_erg_lyric`        | bool  | true    | Enhanced representation guidance for lyrics       |
| `use_erg_diffusion`    | bool  | true    | Enhanced representation guidance during diffusion |
| `guidance_scale_text`  | float | 0.0     | Separate guidance scale for text                  |
| `guidance_scale_lyric` | float | 0.0     | Separate guidance scale for lyrics                |

### Model Customization

| Parameter           | Type   | Default | Description                                                  |
| ------------------- | ------ | ------- | ------------------------------------------------------------ |
| `lora_name_or_path` | string | "none"  | LoRA adapter (e.g., "ACE-Step/ACE-Step-v1-chinese-rap-LoRA") |
| `lora_weight`       | float  | 1.0     | LoRA weight (0.0-2.0)                                        |

### Output Settings

| Parameter       | Type   | Default | Description                           |
| --------------- | ------ | ------- | ------------------------------------- |
| `output_format` | string | "wav"   | Output format: "wav", "mp3", or "ogg" |
| `sample_rate`   | int    | 48000   | Sample rate: 44100 or 48000 Hz        |

## Model Specifications

- **Model Size**: 3.5B parameters
- **Architecture**: Diffusion-based with DCAE and linear transformer
- **Audio Quality**: 48kHz sample rate
- **Generation Speed**: ~15x faster than LLM-based models
- **GPU Requirements**: Optimized for T4/V100/A100 (16GB+ VRAM recommended)
- **Memory Optimization**: CPU offloading enabled for efficient memory usage

## Complete Feature Matrix

| Feature               | Supported | Implementation                                  | Use Case                    |
| --------------------- | --------- | ----------------------------------------------- | --------------------------- |
| **Full-length songs** | ✅ Yes    | Up to 4 minutes in one pass                     | Complete song generation    |
| **Audio fidelity**    | ✅ Yes    | Hybrid diffusion + transformer + DCAE           | High-quality 48kHz output   |
| **Lyric alignment**   | ✅ Yes    | Explicit lyric conditioning with structure tags | Perfect word-music sync     |
| **Continuation**      | ✅ Yes    | Extend at start, end, or both sides             | Song extension workflows    |
| **Inpainting**        | ✅ Yes    | Edit specific time ranges                       | Section replacement/editing |
| **Style transfer**    | ✅ Yes    | Transform existing audio to new styles          | Remix and arrangement       |
| **Voice cloning**     | ✅ Yes    | Audio2audio with reference characteristics      | Apply vocal timbres         |
| **Stem generation**   | ✅ Yes    | Vocal accompaniment generation                  | Backing track creation      |
| **Fast inference**    | ✅ Yes    | ~20 seconds for 4 minutes on A100               | Production-ready speed      |
| **Multi-language**    | ✅ Yes    | 19 languages with top 10 well-supported         | Global music creation       |
| **LoRA adapters**     | ✅ Yes    | Specialized models (e.g., rap, vocal)           | Genre-specific generation   |
| **Advanced guidance** | ✅ Yes    | ERG controls + separate text/lyric guidance     | Fine-grained control        |

## Task-Specific Workflows

### 🎵 Text2Music

- **Purpose**: Generate music from scratch using text prompts and lyrics
- **Best for**: Original composition, full songs, specific genres
- **Key parameters**: `prompt`, `lyrics`, `audio_duration`, `scheduler_type`

### 🔄 Audio2Audio

- **Purpose**: Transform existing audio while preserving structure
- **Best for**: Voice cloning, style adaptation, audio enhancement
- **Key parameters**: `input_audio`, `reference_audio`, `audio2audio_strength`

### ➡️ Continuation

- **Purpose**: Extend existing songs seamlessly
- **Best for**: Making songs longer, adding intros/outros
- **Key parameters**: `input_audio`, `continuation_mode`, `extend_duration`

### 🎨 Inpainting

- **Purpose**: Edit specific sections of existing audio
- **Best for**: Fixing parts, adding elements, lyric changes
- **Key parameters**: `input_audio`, `inpaint_start_time`, `inpaint_end_time`

### 🎭 Style Transfer

- **Purpose**: Transform musical style while keeping structure
- **Best for**: Remixes, genre changes, arrangement variations
- **Key parameters**: `input_audio`, `style_prompt`, `style_strength`

### 🎹 Vocal Accompaniment

- **Purpose**: Generate instrumental backing for vocals
- **Best for**: Creating backing tracks, karaoke versions
- **Key parameters**: `input_audio`, `accompaniment_style`

## Tips for Best Results

### 1. **Prompt Engineering**

- Be specific about genre, mood, and instrumentation
- Use descriptive adjectives (e.g., "upbeat", "melancholic", "energetic")
- Mention specific instruments when desired
- For full songs, describe the overall arc and energy progression

### 2. **Lyrics Structure**

- Use structure tags: `[verse]`, `[chorus]`, `[bridge]`, `[outro]`
- Keep lyrics natural and singable
- Consider rhyme scheme and rhythm
- For longer songs, plan the narrative flow

### 3. **Parameter Tuning**

- Higher `infer_steps` (70-100) for better quality (but slower)
- Adjust `guidance_scale` (15-25) to balance prompt adherence vs. creativity
- Use `seed` for reproducible results
- Try "pingpong" scheduler for the most coherent long-form generation

### 4. **Advanced Techniques**

- Use `variation_strength` for controlled randomness
- Combine style transfer with reference audio for complex transformations
- Apply LoRA adapters for genre-specific results
- Use separate `guidance_scale_text` and `guidance_scale_lyric` for fine control

### 5. **Performance Optimization**

- Start with shorter durations (30-60s) for experimentation
- Use default ERG settings for best quality
- Higher `audio_duration` (120-240s) works best with "pingpong" scheduler
- CPU offloading enables generation on smaller GPUs

## Troubleshooting

### Common Issues

1. **Out of Memory**: The model uses CPU offloading by default, but very long audio or complex prompts may still cause issues
2. **Slow Generation**: Reduce `infer_steps` or `audio_duration` for faster results
3. **Poor Quality**: Increase `infer_steps` or adjust `guidance_scale`

### Error Messages

- **"Model not loaded"**: Ensure the model checkpoint downloaded correctly
- **"Invalid audio duration"**: Keep duration between 10-240 seconds
- **"Tokenization error"**: Check lyrics for unsupported characters

## Support

For issues specific to this Replicate deployment, please check:

- [ACE-Step GitHub Issues](https://github.com/ace-step/ACE-Step/issues)
- [Replicate Documentation](https://replicate.com/docs)
- [Cog Documentation](https://github.com/replicate/cog)

## License

This deployment maintains the Apache 2.0 license of the original ACE-Step project.

## Citation

```bibtex
@misc{gong2025acestep,
    title={ACE-Step: A Step Towards Music Generation Foundation Model},
    author={Junmin Gong, Wenxiao Zhao, Sen Wang, Shengyuan Xu, Jing Guo},
    howpublished={\url{https://github.com/ace-step/ACE-Step}},
    year={2025},
    note={GitHub repository}
}
```
