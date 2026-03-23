# comfyui-id-lora-ltx2.3

ComfyUI custom nodes for **ID-LoRA-2.3** inference — audio+video generation with speaker identity transfer, built on top of LTX-2.3. Supports both **one-stage** (single resolution) and **two-stage** (2x spatial upsampling) pipelines.

stay tuned for native ComfyUI support of ID-LoRA!

## Demo

Two-stage output (max_resolution=1024, HQ mode, 242 frames @ 25fps):

Input voice sample:
[reference.mp3](https://github.com/user-attachments/files/26158470/reference.mp3)


https://github.com/user-attachments/assets/b44879a9-ca1d-4846-bdb7-1142c0b1ccb7



## What it does

ID-LoRA transfers a speaker's vocal identity from a reference audio clip into a generated talking-head video. This package wraps the inference pipelines as ComfyUI nodes:

**One-stage** — generates at a single resolution (lower quality):
```
IDLoraModelLoader --> IDLoraPromptEncoder --> IDLoraOneStageSampler --> SaveVideo
                                                  ^          ^
                                            first_frame  reference_audio
```

**Two-stage** — generates at target resolution then refines at 2x with spatial upsampling (higher quality):
```
IDLoraTwoStageModelLoader --> IDLoraPromptEncoder --> IDLoraTwoStageSampler --> SaveVideo
                                                          ^          ^
                                                    first_frame  reference_audio
```

The two-stage pipeline produces higher-resolution output (e.g. 512x512 -> 1024x1024) by:
1. **Stage 1**: Generating video+audio at target resolution with ID-LoRA + full guidance (CFG, STG, identity guidance, A/V bimodal)
2. **Transition**: Freeing stage-1 models, 2x spatial upsampling the video latent, re-encoding the first frame at 2x
3. **Stage 2**: Refining at 2x resolution with a distilled LoRA only (no guidance), audio frozen from stage 1

An optional **HQ mode** uses the res2s second-order sampler instead of Euler for higher quality.

## Requirements

- **GPU**: NVIDIA GPU with >=24 GB VRAM (48 GB recommended for non-quantized; 80 GB recommended for two-stage non-quantized at high resolutions)
- **Python**: 3.10+
- **ComfyUI**: Recent version with `comfy_api.latest` support
- **Disk**: ~67 GB for one-stage models, ~75 GB for two-stage models (additional upsampler + distilled LoRA)

## Installation

### 0. Install ComfyUI dependencies

If you haven't done so already, install ComfyUI's own requirements first:

```bash
pip install -r ComfyUI/requirements.txt
```

> **Important:** ComfyUI may install `transformers` 5.x, which is incompatible with ID-LoRA. After installing ComfyUI requirements, pin to the latest 4.x release:
> ```bash
> pip install 'transformers>=4.52,<5'
> ```

### 1. Clone ID-LoRA

This package depends on the ID-LoRA repo for its Python packages and model download script. ID-LoRA-2.3 is a subdirectory inside it.

```bash
git clone https://github.com/ID-LoRA/ID-LoRA.git
```

### 2. Download models

Run the download script from the repository root:

```bash
bash ID-LoRA/ID-LoRA-2.3/scripts/download_models.sh models/
```

This downloads all required models from HuggingFace:

| Model | HuggingFace repo | Size |
|-------|-----------------|------|
| LTX-2.3 base checkpoint | `Lightricks/LTX-2.3` | ~44 GB |
| Gemma 3 12B text encoder | `google/gemma-3-12b-it-qat-q4_0-unquantized` | ~23 GB |
| ID-LoRA CelebV-HQ weights | `AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K` | ~1.1 GB |
| ID-LoRA TalkVid weights | `AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K` | ~1.1 GB |

For two-stage, also download (included in the download script):

| Model | HuggingFace repo | Size |
|-------|-----------------|------|
| Spatial upsampler | `Lightricks/LTX-2.3` | ~1 GB |
| Distilled LoRA | `Lightricks/LTX-2.3` | ~7.1 GB |

> **Note:** Gemma requires accepting the license at https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized and logging in with `huggingface-cli login` before downloading.

### 3. Symlink models into ComfyUI

The nodes use ComfyUI's standard model folder system. Create symlinks so ComfyUI can find the downloaded models (adjust the `models/` path if you downloaded elsewhere):

```bash
# Checkpoint
ln -sf "$(pwd)/models/ltx-2.3-22b-dev.safetensors" ComfyUI/models/checkpoints/

# LoRAs
ln -sf "$(pwd)/models/id-lora-celebvhq-ltx2.3/lora_weights.safetensors" ComfyUI/models/loras/id-lora-celebvhq-ltx2.3.safetensors
ln -sf "$(pwd)/models/id-lora-talkvid-ltx2.3/lora_weights.safetensors" ComfyUI/models/loras/id-lora-talkvid-ltx2.3.safetensors
ln -sf "$(pwd)/models/ltx-2.3-22b-distilled-lora-384.safetensors" ComfyUI/models/loras/

# Upscaler
ln -sf "$(pwd)/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" ComfyUI/models/upscale_models/

# Text encoder
ln -sf "$(pwd)/models/gemma-3-12b-it-qat-q4_0-unquantized" ComfyUI/models/text_encoders/
```

If you already have these models in ComfyUI's model directories (e.g. from other LTX nodes), the symlinks are not needed — the nodes will find them automatically via the dropdown selectors.

After setup, ComfyUI's model directories should contain:

```
ComfyUI/models/
├── checkpoints/
│   └── ltx-2.3-22b-dev.safetensors -> .../models/ltx-2.3-22b-dev.safetensors
├── loras/
│   ├── id-lora-celebvhq-ltx2.3.safetensors -> .../models/id-lora-celebvhq-ltx2.3/lora_weights.safetensors
│   ├── id-lora-talkvid-ltx2.3.safetensors -> .../models/id-lora-talkvid-ltx2.3/lora_weights.safetensors
│   └── ltx-2.3-22b-distilled-lora-384.safetensors -> .../models/ltx-2.3-22b-distilled-lora-384.safetensors
├── upscale_models/
│   └── ltx-2.3-spatial-upscaler-x2-1.1.safetensors -> .../models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
└── text_encoders/
    └── gemma-3-12b-it-qat-q4_0-unquantized -> .../models/gemma-3-12b-it-qat-q4_0-unquantized
```

### 4. Install the ltx packages

```bash
pip install -e ID-LoRA/ID-LoRA-2.3/packages/ltx-core
pip install -e ID-LoRA/ID-LoRA-2.3/packages/ltx-pipelines
pip install -e ID-LoRA/ID-LoRA-2.3/packages/ltx-trainer
```

### 5. Clone this repo into ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url> comfyui-id-lora-ltx
```

### 6. Start ComfyUI

```bash
python ComfyUI/main.py
```

Then open http://127.0.0.1:8188 in your browser. The nodes will appear under the **ID-LoRA** category in the node menu.

### Example workflow templates

Two ready-to-use workflows are included in the `example_workflows/` directory:
- **`id_lora_one_stage.json`** — one-stage pipeline
- **`id_lora_two_stage.json`** — two-stage pipeline with 2x upsampling

In ComfyUI, go to **Browse Templates** and look for **comfyui-id-lora-ltx** to load them. They wire all nodes together with LoadImage, LoadAudio, and SaveVideo.

The example workflows use `poster_image.png` (first frame) and `reference.mp3` (speaker reference audio), which are included in the `example_inputs/` directory. Copy them into ComfyUI's `input/` folder so the LoadImage and LoadAudio nodes can find them:

```bash
cp ComfyUI/custom_nodes/ID-LoRA-LTX2.3-ComfyUI/example_inputs/poster_image.png ComfyUI/input/
cp ComfyUI/custom_nodes/ID-LoRA-LTX2.3-ComfyUI/example_inputs/reference.mp3 ComfyUI/input/
```

Then load either workflow template from ComfyUI and click **Queue Prompt** to run.

## Nodes

### ID-LoRA Model Loader

Loads the base LTX-2.3 checkpoint, Gemma text encoder, and ID-LoRA weights into a one-stage pipeline object. This is the slow/expensive node — ComfyUI caches its output when inputs don't change.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `checkpoint_path` | Combo | _(auto-populated)_ | LTX-2.3 checkpoint from `ComfyUI/models/checkpoints/` |
| `text_encoder_path` | String | _(empty = auto-detect)_ | Gemma text encoder directory. Leave empty to auto-detect from `ComfyUI/models/text_encoders/` |
| `lora_path` | Combo | `none` | ID-LoRA `.safetensors` from `ComfyUI/models/loras/`, or `none` to skip |
| `lora_strength` | Float | 1.0 | LoRA strength (0.0-2.0) |
| `quantize` | Combo | `none` | `none`, `int8`, or `fp8` |
| `stg_scale` | Float | 1.0 | Spatio-Temporal Guidance scale (0 disables) |
| `identity_guidance_scale` | Float | 3.0 | Identity guidance strength |
| `av_bimodal_scale` | Float | 3.0 | Audio-video bimodal CFG scale |

**Output**: `ID_LORA_PIPELINE`

### ID-LoRA Two-Stage Model Loader

Loads the base checkpoint, text encoder, ID-LoRA, spatial upsampler, and distilled LoRA for two-stage generation. Like the one-stage loader, ComfyUI caches this node.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `checkpoint_path` | Combo | _(auto-populated)_ | LTX-2.3 checkpoint from `ComfyUI/models/checkpoints/` |
| `text_encoder_path` | String | _(empty = auto-detect)_ | Gemma text encoder directory. Leave empty to auto-detect from `ComfyUI/models/text_encoders/` |
| `lora_path` | Combo | `none` | ID-LoRA `.safetensors` from `ComfyUI/models/loras/`, or `none` to skip |
| `lora_strength` | Float | 1.0 | LoRA strength (0.0-2.0) |
| `upsampler_path` | Combo | _(auto-populated)_ | Spatial upsampler from `ComfyUI/models/upscale_models/` |
| `distilled_lora_path` | Combo | `none` | Distilled LoRA from `ComfyUI/models/loras/`, or `none` to skip |
| `quantize` | Combo | `none` | `none`, `int8`, or `fp8` |
| `stg_scale` | Float | 1.0 | Spatio-Temporal Guidance scale (0 disables) |
| `identity_guidance_scale` | Float | 3.0 | Identity guidance strength |
| `av_bimodal_scale` | Float | 3.0 | Audio-video bimodal CFG scale |

**Output**: `ID_LORA_PIPELINE`

### ID-LoRA Prompt Encoder

Encodes positive and negative text prompts into conditioning tensors. Works with both one-stage and two-stage model loaders. Loads the text encoder temporarily, encodes, then frees it to save VRAM.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `pipeline` | ID_LORA_PIPELINE | — | From either Model Loader |
| `prompt` | String (multiline) | _(empty)_ | Positive prompt |
| `negative_prompt` | String (multiline) | _(default negative)_ | Negative prompt |

**Output**: `ID_LORA_CONDITIONING`

### ID-LoRA One-Stage Sampler

Runs the one-stage generation pipeline: denoising loop with identity guidance, then decodes video and audio. Outputs a ComfyUI `VIDEO` that you wire directly to the built-in **Save Video** node.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `pipeline` | ID_LORA_PIPELINE | — | From Model Loader |
| `conditioning` | ID_LORA_CONDITIONING | — | From Prompt Encoder |
| `first_frame` | IMAGE (optional) | None | First-frame image for face conditioning |
| `reference_audio` | AUDIO (optional) | None | Reference speaker audio for identity transfer |
| `seed` | Int | 42 | Random seed |
| `height` | Int | 512 | Output height (multiple of 32) |
| `width` | Int | 512 | Output width (multiple of 32) |
| `num_frames` | Int | 121 | Number of frames to generate |
| `num_inference_steps` | Int | 30 | Denoising steps |
| `frame_rate` | Float | 25.0 | Output frame rate |
| `video_guidance_scale` | Float | 3.0 | Video CFG scale |
| `audio_guidance_scale` | Float | 7.0 | Audio CFG scale |
| `auto_resolution` | Boolean | True | Auto-detect resolution from first frame, preserving aspect ratio |
| `max_resolution` | Int | 512 | Target long-side resolution when auto_resolution is on — increase for higher res output |

**Output**: `VIDEO` (wire to **Save Video**)

### ID-LoRA Two-Stage Sampler

Runs the two-stage generation pipeline. Stage 1 generates at the specified resolution with full guidance, then stage 2 refines at 2x resolution with the distilled LoRA. Output resolution is **2x** the specified height/width.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `pipeline` | ID_LORA_PIPELINE | — | From Two-Stage Model Loader |
| `conditioning` | ID_LORA_CONDITIONING | — | From Prompt Encoder |
| `first_frame` | IMAGE (optional) | None | First-frame image for face conditioning |
| `reference_audio` | AUDIO (optional) | None | Reference speaker audio for identity transfer |
| `seed` | Int | 42 | Random seed |
| `height` | Int | 512 | Stage 1 height (output = 2x this value) |
| `width` | Int | 512 | Stage 1 width (output = 2x this value) |
| `num_frames` | Int | 121 | Number of frames to generate |
| `num_inference_steps` | Int | 30 | Stage 1 denoising steps (stage 2 uses 3 fixed steps) |
| `frame_rate` | Float | 25.0 | Output frame rate |
| `video_guidance_scale` | Float | 3.0 | Video CFG scale |
| `audio_guidance_scale` | Float | 7.0 | Audio CFG scale |
| `auto_resolution` | Boolean | True | Auto-detect resolution from first frame, preserving aspect ratio |
| `max_resolution` | Int | 512 | Target long-side resolution when auto_resolution is on — increase for higher res output (final output is 2x) |
| `hq_mode` | Boolean | True | Use res2s second-order sampler (higher quality, slower) |

**Output**: `VIDEO` (wire to **Save Video**)

## Prompt format

Prompts use a structured `[VISUAL]` / `[SPEECH]` / `[SOUNDS]` format:

```
[VISUAL]: A medium shot of a young man with curly brown hair, sitting on a beige couch.
He is wearing a light blue shirt and speaking warmly.
[SPEECH]: We are proud to introduce ID-LoRA.
[SOUNDS]: The speaker has a moderate volume and conversational tone. Light instrumental
background music plays softly.
```

All three sections are optional but recommended for best results.

## Example workflows

### One-stage

1. Add **ID-LoRA Model Loader** — set paths to checkpoint, text encoder, and LoRA
2. Add **ID-LoRA Prompt Encoder** — connect the pipeline, write your prompt
3. Add **Load Image** + **Load Audio** — load face image and reference audio
4. Add **ID-LoRA One-Stage Sampler** — connect everything
5. Add **Save Video** — connect the video output

### Two-stage (2x upsampling)

1. Add **ID-LoRA Two-Stage Model Loader** — set paths (including upsampler and distilled LoRA)
2. Add **ID-LoRA Prompt Encoder** — connect the pipeline, write your prompt
3. Add **Load Image** + **Load Audio** — load face image and reference audio
4. Add **ID-LoRA Two-Stage Sampler** — connect everything, optionally enable **hq_mode**
5. Add **Save Video** — connect the video output

The output resolution will be 2x the stage 1 resolution (e.g. 512x512 stage 1 -> 1024x1024 output).

## Reducing memory usage

The full pipeline is memory-intensive — the 22B parameter transformer alone requires significant VRAM. If you run into out-of-memory errors, here are several ways to reduce usage, roughly ordered from least to most impact on quality:

1. **Enable quantization** — Set `quantize` to `int8` or `fp8` on the Model Loader node. This reduces the transformer's memory footprint substantially with a small quality tradeoff. `int8` is recommended as a good balance; `fp8` saves slightly more memory but may introduce more artifacts.

2. **Lower `max_resolution`** — Reducing from 768 to 512 (or lower) significantly reduces VRAM during generation, since memory scales quadratically with resolution. For two-stage, this also reduces the stage-2 resolution proportionally (e.g. 512 → 1024 output instead of 768 → 1536).

3. **Reduce `num_frames`** — Fewer frames means less latent state in memory. Try 121 instead of 242 if memory is tight.

4. **Disable HQ mode** (two-stage only) — The res2s sampler uses more memory than Euler due to its second-order midpoint evaluation. Turning off `hq_mode` saves memory at the cost of some quality.

5. **Use one-stage instead of two-stage** — The two-stage pipeline requires loading two separate transformers sequentially and upsampling the latent to 2x, which has higher peak memory than one-stage.

These options can be combined. For example, `int8` quantization + `max_resolution=512` + one-stage should run on 24 GB GPUs.

## Notes

- **Model paths** use ComfyUI's standard `models/` folder system. Models are selected via dropdown menus that scan `ComfyUI/models/checkpoints/`, `loras/`, `upscale_models/`, and `text_encoders/`. The text encoder path is a string field — leave it empty to auto-detect a Gemma directory from `text_encoders/`.
- **Auto-resolution** (enabled by default) matches the first frame's aspect ratio while capping the long side at `max_resolution` (default 512px). To generate at higher resolution, increase `max_resolution` on the sampler node — the aspect ratio is always preserved. For the two-stage pipeline, this controls stage 1; the final output will be 2x that (e.g. `max_resolution=768` produces up to 1536px output).
- **Two-stage VRAM**: Stage-1 models (transformer, audio encoder) are fully freed before loading stage-2 models. The video encoder is shared for upsampling then freed.
- **HQ mode** (two-stage only): Uses the res2s second-order sampler instead of Euler in both stages for higher quality at the cost of speed.
- **Reusable pipeline** (one-stage only): The video encoder is stashed to CPU (not deleted) after encoding, so re-running with different prompts/seeds doesn't require a full reload.

## File structure

```
comfyui-id-lora-ltx/
├── __init__.py              # Extension entry point (registers all 5 nodes)
├── nodes_model_loader.py    # IDLoraModelLoader + IDLoraTwoStageModelLoader nodes
├── nodes_prompt_encoder.py  # IDLoraPromptEncoder node (shared)
├── nodes_sampler.py         # IDLoraOneStageSampler + IDLoraTwoStageSampler nodes
├── pipeline_wrapper.py      # _IDLoraBase, IDLoraOneStagePipeline, IDLoraTwoStagePipeline
├── example_inputs/
│   ├── poster_image.png         # Example first-frame image
│   └── reference.mp3            # Example reference audio
├── example_outputs/
│   └── two_stage_demo.mp4       # Sample two-stage output
├── example_workflows/
│   ├── id_lora_one_stage.json   # One-stage workflow template
│   └── id_lora_two_stage.json   # Two-stage workflow template
├── pyproject.toml           # Package metadata
├── requirements.txt         # Dependencies
└── README.md
```

## 📝 Citation
```
bibtex
@misc{dahan2026idloraidentitydrivenaudiovideopersonalization,
  title     = {ID-LoRA: Identity-Driven Audio-Video Personalization
               with In-Context LoRA},
  author    = {Aviad Dahan and Moran Yanuka and Noa Kraicer and Lior Wolf and Raja Giryes},
  year      = {2026},
  eprint    = {2603.10256},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url       = {https://arxiv.org/abs/2603.10256}
}
```
