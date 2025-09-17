"""Torch compile optimizations for HyperGen."""

from typing import Any, Optional

import torch
import torch._dynamo
from torch._inductor import config


class OptimizedCompiler:
    """Optimized torch.compile wrapper for diffusion models."""
    
    def __init__(self) -> None:
        """Initialize compiler with optimal settings."""
        self._configure_inductor()
        
    def _configure_inductor(self) -> None:
        """Configure torch inductor for optimal performance."""
        config.coordinate_descent_tuning = True
        config.triton.unique_kernel_names = True
        config.fx_graph_cache = True
        config.max_autotune_gemm_backends = "TRITON"
        
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
    def compile_full(
        self,
        model: torch.nn.Module,
        fullgraph: bool = True,
        dynamic: bool = True,
        mode: str = "max-autotune",
    ) -> torch.nn.Module:
        """Full model compilation.
        
        Args:
            model: Model to compile
            fullgraph: Compile as single graph
            dynamic: Enable dynamic shapes
            mode: Compilation mode
            
        Returns:
            Compiled model
        """
        compile_kwargs = {
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "mode": mode,
        }
        
        if hasattr(model, "compile"):
            model.compile(**compile_kwargs)
        else:
            model = torch.compile(model, **compile_kwargs)
            
        return model
        
    def compile_regional(
        self,
        model: torch.nn.Module,
        fullgraph: bool = True,
        dynamic: bool = True,
        mode: str = "max-autotune",
    ) -> torch.nn.Module:
        """Regional compilation for transformer blocks.
        
        Args:
            model: Model to compile
            fullgraph: Compile blocks as full graphs
            dynamic: Enable dynamic shapes
            mode: Compilation mode
            
        Returns:
            Compiled model
        """
        if hasattr(model, "compile_repeated_blocks"):
            model.compile_repeated_blocks(
                fullgraph=fullgraph,
                dynamic=dynamic,
                mode=mode,
            )
        else:
            self._compile_transformer_blocks(
                model,
                fullgraph=fullgraph,
                dynamic=dynamic,
                mode=mode,
            )
            
        return model
        
    def _compile_transformer_blocks(
        self,
        model: torch.nn.Module,
        fullgraph: bool = True,
        dynamic: bool = True,
        mode: str = "max-autotune",
    ) -> None:
        """Compile individual transformer blocks.
        
        Args:
            model: Model with transformer blocks
            fullgraph: Compile blocks as full graphs
            dynamic: Enable dynamic shapes
            mode: Compilation mode
        """
        compile_kwargs = {
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "mode": mode,
        }
        
        if hasattr(model, "transformer_blocks"):
            blocks = model.transformer_blocks
        elif hasattr(model, "blocks"):
            blocks = model.blocks
        elif hasattr(model, "layers"):
            blocks = model.layers
        else:
            blocks = []
            
        if blocks and len(blocks) > 0:
            first_block = blocks[0]
            compiled_block = torch.compile(first_block, **compile_kwargs)
            
            for i, block in enumerate(blocks):
                if i == 0:
                    blocks[i] = compiled_block
                else:
                    blocks[i].load_state_dict(compiled_block.state_dict())
                    
    def compile_with_cache(
        self,
        model: torch.nn.Module,
        cache_dir: Optional[str] = None,
        **compile_kwargs: Any,
    ) -> torch.nn.Module:
        """Compile with persistent cache.
        
        Args:
            model: Model to compile
            cache_dir: Cache directory
            **compile_kwargs: Compilation arguments
            
        Returns:
            Compiled model
        """
        if cache_dir:
            torch._inductor.config.fx_graph_cache_dir = cache_dir
            
        return self.compile_full(model, **compile_kwargs)
        
    def clear_compile_cache(self) -> None:
        """Clear compilation cache."""
        torch._dynamo.reset()
        torch._inductor.codecache.FxGraphCache.clear()
        
    def benchmark_compilation(
        self,
        model: torch.nn.Module,
        input_shape: tuple[int, ...],
        num_warmup: int = 3,
        num_runs: int = 10,
    ) -> dict[str, float]:
        """Benchmark compilation performance.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        import time
        
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        dummy_input = torch.randn(input_shape, device=device, dtype=dtype)
        
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(dummy_input)
                
        torch.cuda.synchronize()
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = model(dummy_input)
                
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append(end - start)
            
        return {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "speedup": times[0] / (sum(times) / len(times)),
        }