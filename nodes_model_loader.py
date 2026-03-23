"""IDLoraModelLoader nodes — loads checkpoint + text encoder + LoRA into pipeline."""

from __future__ import annotations

import os

import folder_paths
import torch
from comfy_api.latest import io
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP

from .pipeline_wrapper import IDLoraOneStagePipeline, IDLoraTwoStagePipeline, IDLoraPipelineType


def _resolve_text_encoder_path(p: str) -> str:
    """Resolve a text-encoder directory path via ComfyUI's folder_paths system.

    - Empty string: auto-detect by scanning text_encoders folders for a
      subdirectory whose name contains 'gemma'.
    - Relative path: look inside ComfyUI's text_encoders folders.
    - Absolute path: pass through unchanged.
    """
    if not p:
        for base in folder_paths.get_folder_paths("text_encoders"):
            try:
                for entry in os.scandir(base):
                    if entry.is_dir() and "gemma" in entry.name.lower():
                        return entry.path
            except OSError:
                continue
        return p
    if os.path.isabs(p):
        return p
    # Relative — look in each text_encoders folder
    for base in folder_paths.get_folder_paths("text_encoders"):
        candidate = os.path.join(base, p)
        if os.path.isdir(candidate):
            return candidate
    return p


class IDLoraModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IDLoraModelLoader",
            display_name="ID-LoRA Model Loader",
            category="ID-LoRA",
            description="Load LTX-2.3 checkpoint, text encoder, and ID-LoRA weights into a reusable pipeline.",
            inputs=[
                io.Combo.Input("checkpoint_path", options=folder_paths.get_filename_list("checkpoints"),
                               tooltip="LTX-2.3 base checkpoint (.safetensors)."),
                io.String.Input("text_encoder_path", default="",
                                tooltip="Gemma text-encoder directory. Leave empty to auto-detect from ComfyUI's text_encoders folders."),
                io.Combo.Input("lora_path", options=["none"] + folder_paths.get_filename_list("loras"),
                               tooltip="ID-LoRA checkpoint (.safetensors). Select 'none' to skip."),
                io.Float.Input("lora_strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip="LoRA application strength."),
                io.Combo.Input("quantize", options=["none", "int8", "fp8"],
                               tooltip="Quantization mode for the transformer."),
                io.Float.Input("stg_scale", default=1.0, min=0.0, max=10.0, step=0.1,
                               tooltip="STG (Spatio-Temporal Guidance) scale. 0 disables."),
                io.Float.Input("identity_guidance_scale", default=3.0, min=0.0, max=20.0, step=0.1,
                               tooltip="Identity guidance scale for speaker transfer."),
                io.Float.Input("av_bimodal_scale", default=3.0, min=0.0, max=20.0, step=0.1,
                               tooltip="Audio-video bimodal CFG scale."),
            ],
            outputs=[
                io.Custom("ID_LORA_PIPELINE").Output(display_name="Pipeline", tooltip="Loaded ID-LoRA pipeline."),
            ],
        )

    @classmethod
    def execute(
        cls,
        checkpoint_path: str,
        text_encoder_path: str,
        lora_path: str,
        lora_strength: float,
        quantize: str,
        stg_scale: float,
        identity_guidance_scale: float,
        av_bimodal_scale: float,
    ) -> io.NodeOutput:
        device = torch.device("cuda")
        checkpoint_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint_path)
        text_encoder_path = _resolve_text_encoder_path(text_encoder_path)

        loras = []
        if lora_path != "none":
            loras.append(LoraPathStrengthAndSDOps(
                path=folder_paths.get_full_path_or_raise("loras", lora_path),
                strength=lora_strength,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            ))

        pipeline = IDLoraOneStagePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=text_encoder_path,
            loras=loras,
            device=device,
            quantize=(quantize == "int8"),
            fp8=(quantize == "fp8"),
            stg_scale=stg_scale,
            identity_guidance=True,
            identity_guidance_scale=identity_guidance_scale,
            av_bimodal_cfg=True,
            av_bimodal_scale=av_bimodal_scale,
        )

        # NOTE: load_models() is NOT called here. The original ID-LoRA script
        # encodes prompts BEFORE loading the heavy models (transformer, VAEs)
        # to avoid GPU memory pressure. The sampler node calls load_models()
        # after prompt encoding is complete.
        return io.NodeOutput(pipeline)


class IDLoraTwoStageModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IDLoraTwoStageModelLoader",
            display_name="ID-LoRA Two-Stage Model Loader",
            category="ID-LoRA",
            description="Load LTX-2.3 checkpoint, text encoder, ID-LoRA, upsampler, and distilled LoRA for two-stage generation.",
            inputs=[
                io.Combo.Input("checkpoint_path", options=folder_paths.get_filename_list("checkpoints"),
                               tooltip="LTX-2.3 base checkpoint (.safetensors)."),
                io.String.Input("text_encoder_path", default="",
                                tooltip="Gemma text-encoder directory. Leave empty to auto-detect from ComfyUI's text_encoders folders."),
                io.Combo.Input("lora_path", options=["none"] + folder_paths.get_filename_list("loras"),
                               tooltip="ID-LoRA checkpoint (.safetensors). Select 'none' to skip."),
                io.Float.Input("lora_strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip="LoRA application strength."),
                io.Combo.Input("upsampler_path", options=folder_paths.get_filename_list("upscale_models"),
                               tooltip="Spatial upsampler checkpoint (.safetensors)."),
                io.Combo.Input("distilled_lora_path", options=["none"] + folder_paths.get_filename_list("loras"),
                               tooltip="Distilled LoRA for stage 2 (.safetensors). Select 'none' to skip."),
                io.Combo.Input("quantize", options=["none", "int8", "fp8"],
                               tooltip="Quantization mode for the transformer."),
                io.Float.Input("stg_scale", default=1.0, min=0.0, max=10.0, step=0.1,
                               tooltip="STG (Spatio-Temporal Guidance) scale. 0 disables."),
                io.Float.Input("identity_guidance_scale", default=3.0, min=0.0, max=20.0, step=0.1,
                               tooltip="Identity guidance scale for speaker transfer."),
                io.Float.Input("av_bimodal_scale", default=3.0, min=0.0, max=20.0, step=0.1,
                               tooltip="Audio-video bimodal CFG scale."),
            ],
            outputs=[
                io.Custom("ID_LORA_PIPELINE").Output(display_name="Pipeline", tooltip="Loaded ID-LoRA two-stage pipeline."),
            ],
        )

    @classmethod
    def execute(
        cls,
        checkpoint_path: str,
        text_encoder_path: str,
        lora_path: str,
        lora_strength: float,
        upsampler_path: str,
        distilled_lora_path: str,
        quantize: str,
        stg_scale: float,
        identity_guidance_scale: float,
        av_bimodal_scale: float,
    ) -> io.NodeOutput:
        device = torch.device("cuda")
        checkpoint_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint_path)
        text_encoder_path = _resolve_text_encoder_path(text_encoder_path)
        upsampler_path = folder_paths.get_full_path_or_raise("upscale_models", upsampler_path)

        ic_loras = []
        if lora_path != "none":
            ic_loras.append(LoraPathStrengthAndSDOps(
                path=folder_paths.get_full_path_or_raise("loras", lora_path),
                strength=lora_strength,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            ))

        distilled_lora_resolved = None
        if distilled_lora_path != "none":
            distilled_lora_resolved = folder_paths.get_full_path_or_raise("loras", distilled_lora_path)

        pipeline = IDLoraTwoStagePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=text_encoder_path,
            upsampler_path=upsampler_path,
            distilled_lora_path=distilled_lora_resolved,
            ic_loras=ic_loras,
            device=device,
            quantize=(quantize == "int8"),
            fp8=(quantize == "fp8"),
            stg_scale=stg_scale,
            identity_guidance=True,
            identity_guidance_scale=identity_guidance_scale,
            av_bimodal_cfg=True,
            av_bimodal_scale=av_bimodal_scale,
        )

        # NOTE: load_stage_1_models() is NOT called here — deferred to sampler
        # node after prompt encoding is complete, to minimize peak VRAM.
        return io.NodeOutput(pipeline)
