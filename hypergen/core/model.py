"""Model management and inference for HyperGen."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import (
    FluxPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image
from transformers import AutoTokenizer

from hypergen.core.dataset import Dataset
from hypergen.core.lora import LoRAAdapter, LoRATrainer
from hypergen.optimizations.compiler import OptimizedCompiler
from hypergen.optimizations.memory import MemoryOptimizer


class Model:
    """Unified model interface for diffusion models with optimizations."""
    
    _model_registry: Dict[str, type] = {
        "flux": FluxPipeline,
        "sdxl": StableDiffusionXLPipeline,
        "sd": StableDiffusionPipeline,
    }
    
    def __init__(
        self,
        pipeline: Any,
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize model with pipeline."""
        self.pipeline = pipeline
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.compiler = OptimizedCompiler()
        self.memory_optimizer = MemoryOptimizer()
        self._lora_adapters: Dict[str, LoRAAdapter] = {}
        self._active_lora: Optional[str] = None
        self._compiled = False
        
    @classmethod
    def load(
        cls,
        model_id: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        enable_cpu_offload: bool = False,
        enable_quantization: bool = False,
        quantization_bits: int = 4,
        compile_mode: str = "regional",
    ) -> Model:
        """Load a model with automatic optimizations.
        
        Args:
            model_id: Model identifier or HuggingFace model path
            device: Device to load model on
            dtype: Model dtype (defaults to bfloat16 for modern GPUs)
            enable_cpu_offload: Enable CPU offloading for low VRAM
            enable_quantization: Enable weight quantization
            quantization_bits: Bits for quantization (4 or 8)
            compile_mode: Compilation mode - "none", "regional", or "full"
            
        Returns:
            Optimized Model instance
        """
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
        model_type = cls._detect_model_type(model_id)
        pipeline_class = cls._model_registry.get(model_type)
        
        if pipeline_class is None:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        pipeline = pipeline_class.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        
        if device == "cuda" and torch.cuda.is_available():
            pipeline = pipeline.to(device)
            
        model = cls(pipeline, model_id, device, dtype)
        
        if enable_cpu_offload:
            model.enable_cpu_offload()
            
        if enable_quantization:
            model.enable_quantization(bits=quantization_bits)
            
        if compile_mode != "none":
            model.compile(mode=compile_mode)
            
        return model
        
    @staticmethod
    def _detect_model_type(model_id: str) -> str:
        """Detect model type from model ID."""
        model_id_lower = model_id.lower()
        if "flux" in model_id_lower:
            return "flux"
        elif "sdxl" in model_id_lower:
            return "sdxl"
        elif "stable-diffusion" in model_id_lower:
            return "sd"
        else:
            return "flux"
            
    def compile(
        self,
        mode: str = "regional",
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> None:
        """Apply torch.compile optimizations.
        
        Args:
            mode: Compilation mode - "regional" or "full"
            fullgraph: Whether to compile as full graph
            dynamic: Enable dynamic shapes to avoid recompilation
        """
        if self._compiled:
            return
            
        if hasattr(self.pipeline, "transformer"):
            target = self.pipeline.transformer
        elif hasattr(self.pipeline, "unet"):
            target = self.pipeline.unet
        else:
            raise ValueError("No compilable component found in pipeline")
            
        if mode == "regional":
            self.compiler.compile_regional(target, fullgraph=fullgraph, dynamic=dynamic)
        elif mode == "full":
            self.compiler.compile_full(target, fullgraph=fullgraph, dynamic=dynamic)
        else:
            raise ValueError(f"Invalid compilation mode: {mode}")
            
        self._compiled = True
        
    def enable_cpu_offload(self) -> None:
        """Enable CPU offloading to reduce VRAM usage."""
        self.memory_optimizer.enable_cpu_offload(self.pipeline)
        
    def enable_quantization(self, bits: int = 4) -> None:
        """Enable weight quantization.
        
        Args:
            bits: Number of bits for quantization (4 or 8)
        """
        self.memory_optimizer.quantize_model(self.pipeline, bits=bits)
        
    def train_lora(
        self,
        dataset: Dataset,
        rank: int = 16,
        learning_rate: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        output_dir: Optional[str] = None,
    ) -> LoRAAdapter:
        """Train a LoRA adapter on the model.
        
        Args:
            dataset: Training dataset
            rank: LoRA rank
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Training batch size
            gradient_accumulation_steps: Gradient accumulation steps
            output_dir: Directory to save LoRA weights
            
        Returns:
            Trained LoRA adapter
        """
        trainer = LoRATrainer(self)
        adapter = trainer.train(
            dataset=dataset,
            rank=rank,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            output_dir=output_dir,
        )
        
        adapter_name = f"lora_{len(self._lora_adapters)}"
        self._lora_adapters[adapter_name] = adapter
        
        return adapter
        
    def load_lora(
        self,
        path: Union[str, Path],
        adapter_name: Optional[str] = None,
        hotswap: bool = False,
    ) -> None:
        """Load a LoRA adapter.
        
        Args:
            path: Path to LoRA weights
            adapter_name: Name for the adapter
            hotswap: Enable hotswapping without recompilation
        """
        if adapter_name is None:
            adapter_name = f"lora_{len(self._lora_adapters)}"
            
        if hotswap and self._compiled:
            self.pipeline.load_lora_weights(path, adapter_name=adapter_name, hotswap=True)
        else:
            self.pipeline.load_lora_weights(path, adapter_name=adapter_name)
            
        self._active_lora = adapter_name
        
    def enable_lora_hotswap(self, max_rank: int = 64) -> None:
        """Enable LoRA hotswapping to avoid recompilation.
        
        Args:
            max_rank: Maximum LoRA rank across all adapters
        """
        if hasattr(self.pipeline, "enable_lora_hotswap"):
            self.pipeline.enable_lora_hotswap(target_rank=max_rank)
            
    def run(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        lora: Optional[Union[str, LoRAAdapter]] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Run inference with the model.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for CFG
            width: Image width
            height: Image height
            seed: Random seed
            lora: LoRA adapter to use
            **kwargs: Additional pipeline arguments
            
        Returns:
            Generated image
        """
        if lora is not None:
            if isinstance(lora, str):
                self.load_lora(lora)
            elif isinstance(lora, LoRAAdapter):
                self.load_lora(lora.path)
                
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        pipeline_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        
        if negative_prompt:
            pipeline_kwargs["negative_prompt"] = negative_prompt
            
        if width and height:
            pipeline_kwargs["width"] = width
            pipeline_kwargs["height"] = height
            
        pipeline_kwargs.update(kwargs)
        
        with torch.inference_mode():
            output = self.pipeline(**pipeline_kwargs)
            
        return output.images[0]
        
    def run_lora(
        self,
        prompt: str,
        lora: Union[str, LoRAAdapter],
        **kwargs: Any,
    ) -> Image.Image:
        """Convenience method to run inference with a specific LoRA.
        
        Args:
            prompt: Text prompt
            lora: LoRA adapter or path
            **kwargs: Additional arguments for run()
            
        Returns:
            Generated image
        """
        return self.run(prompt=prompt, lora=lora, **kwargs)
        
    def clear_cache(self) -> None:
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    def benchmark(
        self,
        prompt: str = "A cat holding a sign that says hello world",
        num_runs: int = 5,
        warmup_runs: int = 2,
    ) -> Dict[str, float]:
        """Benchmark model performance.
        
        Args:
            prompt: Test prompt
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results
        """
        import time
        
        for _ in range(warmup_runs):
            self.run(prompt)
            self.clear_cache()
            
        times = []
        memory_peaks = []
        
        for _ in range(num_runs):
            torch.cuda.reset_peak_memory_stats()
            
            start = time.perf_counter()
            self.run(prompt)
            end = time.perf_counter()
            
            times.append(end - start)
            memory_peaks.append(torch.cuda.max_memory_allocated() / 1e9)
            
            self.clear_cache()
            
        return {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "mean_memory_gb": sum(memory_peaks) / len(memory_peaks),
            "max_memory_gb": max(memory_peaks),
        }