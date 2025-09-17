"""LoRA training and management for HyperGen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

if TYPE_CHECKING:
    from hypergen.core.dataset import Dataset
    from hypergen.core.model import Model


class LoRAAdapter:
    """LoRA adapter wrapper."""
    
    def __init__(
        self,
        path: Path,
        rank: int,
        alpha: float,
        target_modules: list[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize LoRA adapter.
        
        Args:
            path: Path to saved LoRA weights
            rank: LoRA rank
            alpha: LoRA alpha scaling factor
            target_modules: Target modules for LoRA
            metadata: Optional metadata
        """
        self.path = path
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules
        self.metadata = metadata or {}
        
    def save(self, path: Optional[Path] = None) -> None:
        """Save LoRA adapter metadata.
        
        Args:
            path: Optional save path
        """
        save_path = path or self.path
        metadata_path = save_path / "adapter_metadata.json"
        
        with open(metadata_path, "w") as f:
            json.dump({
                "rank": self.rank,
                "alpha": self.alpha,
                "target_modules": self.target_modules,
                **self.metadata,
            }, f, indent=2)
            
    @classmethod
    def load(cls, path: Path) -> LoRAAdapter:
        """Load LoRA adapter from path.
        
        Args:
            path: Path to LoRA weights
            
        Returns:
            LoRA adapter instance
        """
        metadata_path = path / "adapter_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {"rank": 16, "alpha": 16, "target_modules": []}
            
        return cls(
            path=path,
            rank=metadata.pop("rank", 16),
            alpha=metadata.pop("alpha", 16),
            target_modules=metadata.pop("target_modules", []),
            metadata=metadata,
        )


class LoRATrainer:
    """LoRA training implementation."""
    
    def __init__(self, model: Model) -> None:
        """Initialize LoRA trainer.
        
        Args:
            model: HyperGen model instance
        """
        self.model = model
        self.accelerator = Accelerator(
            mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
            gradient_accumulation_steps=1,
        )
        
    def train(
        self,
        dataset: Dataset,
        rank: int = 16,
        alpha: Optional[float] = None,
        target_modules: Optional[list[str]] = None,
        learning_rate: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        output_dir: Optional[str] = None,
        validation_dataset: Optional[Dataset] = None,
        save_steps: int = 500,
        eval_steps: int = 100,
    ) -> LoRAAdapter:
        """Train LoRA adapter.
        
        Args:
            dataset: Training dataset
            rank: LoRA rank
            alpha: LoRA alpha (defaults to rank)
            target_modules: Target modules for LoRA
            learning_rate: Learning rate
            epochs: Number of epochs
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation
            output_dir: Output directory for weights
            validation_dataset: Optional validation dataset
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            
        Returns:
            Trained LoRA adapter
        """
        if alpha is None:
            alpha = float(rank)
            
        if target_modules is None:
            target_modules = self._get_default_target_modules()
            
        if output_dir is None:
            output_dir = Path(f"lora_checkpoints/{self.model.model_id.replace('/', '_')}")
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.accelerator.gradient_accumulation_steps = gradient_accumulation_steps
        
        trainable_model = self._prepare_model_for_training(
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
        )
        
        dataloader = dataset.create_dataloader(
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        
        optimizer = AdamW(
            trainable_model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        
        num_training_steps = len(dataloader) * epochs // gradient_accumulation_steps
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=learning_rate * 0.1,
        )
        
        trainable_model, optimizer, dataloader, scheduler = self.accelerator.prepare(
            trainable_model, optimizer, dataloader, scheduler
        )
        
        global_step = 0
        best_loss = float("inf")
        
        for epoch in range(epochs):
            trainable_model.train()
            epoch_losses = []
            
            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                disable=not self.accelerator.is_local_main_process,
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(trainable_model):
                    loss = self._compute_loss(trainable_model, batch)
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(trainable_model.parameters(), 1.0)
                        
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    epoch_losses.append(loss.item())
                    
                if batch_idx % gradient_accumulation_steps == 0:
                    global_step += 1
                    
                    if global_step % save_steps == 0:
                        self._save_checkpoint(
                            trainable_model,
                            output_dir / f"checkpoint-{global_step}",
                            rank,
                            alpha,
                            target_modules,
                        )
                        
                    if validation_dataset and global_step % eval_steps == 0:
                        val_loss = self._evaluate(trainable_model, validation_dataset)
                        if val_loss < best_loss:
                            best_loss = val_loss
                            self._save_checkpoint(
                                trainable_model,
                                output_dir / "best",
                                rank,
                                alpha,
                                target_modules,
                            )
                            
                progress_bar.set_postfix({"loss": sum(epoch_losses) / len(epoch_losses)})
                
        self._save_checkpoint(
            trainable_model,
            output_dir / "final",
            rank,
            alpha,
            target_modules,
        )
        
        return LoRAAdapter(
            path=output_dir / "final",
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            metadata={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
        )
        
    def _get_default_target_modules(self) -> list[str]:
        """Get default target modules based on model type."""
        if hasattr(self.model.pipeline, "transformer"):
            return [
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "add_out_proj.0",
            ]
        elif hasattr(self.model.pipeline, "unet"):
            return [
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "conv_in",
                "conv_out",
            ]
        else:
            raise ValueError("Unknown model architecture")
            
    def _prepare_model_for_training(
        self,
        rank: int,
        alpha: float,
        target_modules: list[str],
    ) -> nn.Module:
        """Prepare model for LoRA training."""
        if hasattr(self.model.pipeline, "transformer"):
            base_model = self.model.pipeline.transformer
        elif hasattr(self.model.pipeline, "unet"):
            base_model = self.model.pipeline.unet
        else:
            raise ValueError("No trainable component found")
            
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.DIFFUSION,
        )
        
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        
        return model
        
    def _compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute training loss."""
        if self.model.dataset_type == "image":
            images = batch["image"]
            captions = batch["caption"]
            
            latents = self._encode_images(images)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, 1000, (latents.shape[0],), device=latents.device
            )
            
            noisy_latents = self._add_noise(latents, noise, timesteps)
            
            if hasattr(self.model.pipeline, "text_encoder"):
                encoder_hidden_states = self._encode_text(captions)
            else:
                encoder_hidden_states = None
                
            model_pred = model(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            
            loss = nn.functional.mse_loss(model_pred, noise)
            
        else:
            raise NotImplementedError("Video training not yet implemented")
            
        return loss
        
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latents."""
        if hasattr(self.model.pipeline, "vae"):
            with torch.no_grad():
                latents = self.model.pipeline.vae.encode(images).latent_dist.sample()
                latents = latents * self.model.pipeline.vae.config.scaling_factor
        else:
            latents = images
            
        return latents
        
    def _encode_text(self, captions: list[str]) -> torch.Tensor:
        """Encode text to embeddings."""
        if hasattr(self.model.pipeline, "tokenizer") and hasattr(self.model.pipeline, "text_encoder"):
            inputs = self.model.pipeline.tokenizer(
                captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            with torch.no_grad():
                encoder_hidden_states = self.model.pipeline.text_encoder(
                    inputs.input_ids
                )[0]
        else:
            encoder_hidden_states = torch.zeros(
                (len(captions), 77, 768),
                device=self.model.device,
                dtype=self.model.dtype,
            )
            
        return encoder_hidden_states
        
    def _add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to latents for diffusion training."""
        from diffusers import DDPMScheduler
        
        scheduler = DDPMScheduler.from_pretrained(
            self.model.model_id,
            subfolder="scheduler",
        )
        
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        return noisy_latents
        
    def _evaluate(
        self,
        model: nn.Module,
        dataset: Dataset,
    ) -> float:
        """Evaluate model on validation set."""
        model.eval()
        val_losses = []
        
        dataloader = dataset.create_dataloader(
            batch_size=1,
            shuffle=False,
        )
        
        with torch.no_grad():
            for batch in dataloader:
                loss = self._compute_loss(model, batch)
                val_losses.append(loss.item())
                
        model.train()
        
        return sum(val_losses) / len(val_losses)
        
    def _save_checkpoint(
        self,
        model: nn.Module,
        path: Path,
        rank: int,
        alpha: float,
        target_modules: list[str],
    ) -> None:
        """Save training checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(path)
            
            adapter = LoRAAdapter(
                path=path,
                rank=rank,
                alpha=alpha,
                target_modules=target_modules,
            )
            adapter.save()