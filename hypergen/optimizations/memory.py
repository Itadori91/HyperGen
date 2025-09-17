"""Memory optimization utilities for HyperGen."""

from typing import Any, Optional

import torch
import torch.nn as nn


class MemoryOptimizer:
    """Memory optimization techniques for diffusion models."""
    
    def __init__(self) -> None:
        """Initialize memory optimizer."""
        self._original_modules: dict[str, nn.Module] = {}
        
    def enable_cpu_offload(
        self,
        pipeline: Any,
        gpu_id: int = 0,
    ) -> None:
        """Enable CPU offloading for pipeline components.
        
        Args:
            pipeline: Diffusion pipeline
            gpu_id: GPU device ID
        """
        if hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            self._manual_cpu_offload(pipeline, gpu_id)
            
    def _manual_cpu_offload(
        self,
        pipeline: Any,
        gpu_id: int = 0,
    ) -> None:
        """Manual CPU offloading implementation.
        
        Args:
            pipeline: Diffusion pipeline
            gpu_id: GPU device ID
        """
        device = torch.device(f"cuda:{gpu_id}")
        
        components = ["text_encoder", "text_encoder_2", "transformer", "unet", "vae"]
        
        for component_name in components:
            if hasattr(pipeline, component_name):
                component = getattr(pipeline, component_name)
                if component is not None:
                    self._setup_hooks_for_offload(component, device)
                    
    def _setup_hooks_for_offload(
        self,
        module: nn.Module,
        device: torch.device,
    ) -> None:
        """Setup forward hooks for CPU offloading.
        
        Args:
            module: Module to offload
            device: Target GPU device
        """
        def pre_forward_hook(module: nn.Module, args: Any) -> None:
            module.to(device)
            
        def post_forward_hook(
            module: nn.Module,
            args: Any,
            output: Any,
        ) -> None:
            module.to("cpu")
            torch.cuda.empty_cache()
            
        module.register_forward_pre_hook(pre_forward_hook)
        module.register_forward_hook(post_forward_hook)
        
    def quantize_model(
        self,
        pipeline: Any,
        bits: int = 4,
        quantization_config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Quantize model weights to reduce memory.
        
        Args:
            pipeline: Diffusion pipeline
            bits: Number of bits for quantization
            quantization_config: Custom quantization config
        """
        try:
            from transformers import BitsAndBytesConfig
            
            if quantization_config is None:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=(bits == 4),
                    load_in_8bit=(bits == 8),
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                
            components = ["transformer", "unet", "text_encoder", "text_encoder_2"]
            
            for component_name in components:
                if hasattr(pipeline, component_name):
                    component = getattr(pipeline, component_name)
                    if component is not None:
                        self._quantize_component(component, quantization_config)
                        
        except ImportError:
            self._fallback_quantization(pipeline, bits)
            
    def _quantize_component(
        self,
        component: nn.Module,
        config: Any,
    ) -> None:
        """Quantize individual component.
        
        Args:
            component: Component to quantize
            config: Quantization config
        """
        try:
            from bitsandbytes.nn import Linear4bit, Linear8bitLt
            
            target_class = Linear4bit if config.load_in_4bit else Linear8bitLt
            
            for name, module in component.named_modules():
                if isinstance(module, nn.Linear):
                    if config.load_in_4bit:
                        quantized = Linear4bit(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            compute_dtype=config.bnb_4bit_compute_dtype,
                            compress_statistics=config.bnb_4bit_use_double_quant,
                            quant_type=config.bnb_4bit_quant_type,
                        )
                    else:
                        quantized = Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                        )
                        
                    quantized.weight.data = module.weight.data
                    if module.bias is not None:
                        quantized.bias.data = module.bias.data
                        
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = component
                    
                    for part in parent_name.split("."):
                        if part:
                            parent = getattr(parent, part)
                            
                    setattr(parent, child_name, quantized)
                    
        except ImportError:
            pass
            
    def _fallback_quantization(
        self,
        pipeline: Any,
        bits: int,
    ) -> None:
        """Fallback quantization using PyTorch native methods.
        
        Args:
            pipeline: Diffusion pipeline
            bits: Number of bits
        """
        components = ["transformer", "unet", "vae"]
        
        for component_name in components:
            if hasattr(pipeline, component_name):
                component = getattr(pipeline, component_name)
                if component is not None:
                    self._apply_dynamic_quantization(component, bits)
                    
    def _apply_dynamic_quantization(
        self,
        module: nn.Module,
        bits: int,
    ) -> None:
        """Apply PyTorch dynamic quantization.
        
        Args:
            module: Module to quantize
            bits: Number of bits
        """
        if bits == 8:
            torch.quantization.quantize_dynamic(
                module,
                {nn.Linear},
                dtype=torch.qint8,
            )
            
    def enable_gradient_checkpointing(
        self,
        pipeline: Any,
    ) -> None:
        """Enable gradient checkpointing to save memory during training.
        
        Args:
            pipeline: Diffusion pipeline
        """
        components = ["transformer", "unet", "text_encoder"]
        
        for component_name in components:
            if hasattr(pipeline, component_name):
                component = getattr(pipeline, component_name)
                if component is not None and hasattr(component, "enable_gradient_checkpointing"):
                    component.enable_gradient_checkpointing()
                    
    def enable_attention_slicing(
        self,
        pipeline: Any,
        slice_size: Optional[int] = None,
    ) -> None:
        """Enable attention slicing to reduce memory usage.
        
        Args:
            pipeline: Diffusion pipeline
            slice_size: Slice size for attention
        """
        if hasattr(pipeline, "enable_attention_slicing"):
            if slice_size is None:
                pipeline.enable_attention_slicing("auto")
            else:
                pipeline.enable_attention_slicing(slice_size)
                
    def enable_vae_slicing(
        self,
        pipeline: Any,
    ) -> None:
        """Enable VAE slicing for decoding.
        
        Args:
            pipeline: Diffusion pipeline
        """
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
            
    def enable_xformers(
        self,
        pipeline: Any,
    ) -> None:
        """Enable xFormers memory efficient attention.
        
        Args:
            pipeline: Diffusion pipeline
        """
        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
                
    def optimize_for_inference(
        self,
        pipeline: Any,
        enable_cpu_offload: bool = True,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        enable_xformers: bool = True,
    ) -> None:
        """Apply all inference optimizations.
        
        Args:
            pipeline: Diffusion pipeline
            enable_cpu_offload: Enable CPU offloading
            enable_attention_slicing: Enable attention slicing
            enable_vae_slicing: Enable VAE slicing
            enable_xformers: Enable xFormers
        """
        if enable_cpu_offload:
            self.enable_cpu_offload(pipeline)
            
        if enable_attention_slicing:
            self.enable_attention_slicing(pipeline)
            
        if enable_vae_slicing:
            self.enable_vae_slicing(pipeline)
            
        if enable_xformers:
            self.enable_xformers(pipeline)
            
    def get_memory_stats(self) -> dict[str, float]:
        """Get current GPU memory statistics.
        
        Returns:
            Memory statistics in GB
        """
        if not torch.cuda.is_available():
            return {}
            
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "free_gb": (torch.cuda.mem_get_info()[0]) / 1e9,
            "total_gb": (torch.cuda.mem_get_info()[1]) / 1e9,
        }