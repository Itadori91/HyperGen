"""Dataset management for HyperGen training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms


class ImageDataset(TorchDataset):
    """Dataset for image-caption pairs."""
    
    def __init__(
        self,
        image_paths: List[Path],
        captions: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize image dataset.
        
        Args:
            image_paths: List of image file paths
            captions: Optional list of captions
            transform: Image transformations
        """
        self.image_paths = image_paths
        self.captions = captions or [""] * len(image_paths)
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "caption": self.captions[idx],
            "path": str(self.image_paths[idx]),
        }


class VideoDataset(TorchDataset):
    """Dataset for video-caption pairs."""
    
    def __init__(
        self,
        video_paths: List[Path],
        captions: Optional[List[str]] = None,
        num_frames: int = 16,
        frame_stride: int = 2,
    ) -> None:
        """Initialize video dataset.
        
        Args:
            video_paths: List of video file paths
            captions: Optional list of captions
            num_frames: Number of frames to extract
            frame_stride: Stride between frames
        """
        self.video_paths = video_paths
        self.captions = captions or [""] * len(video_paths)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.video_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        import cv2
        
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(str(video_path))
        
        frames = []
        frame_idx = 0
        
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % self.frame_stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
                
            frame_idx += 1
            
        cap.release()
        
        if len(frames) < self.num_frames:
            frames.extend([frames[-1]] * (self.num_frames - len(frames)))
            
        return {
            "frames": frames,
            "caption": self.captions[idx],
            "path": str(video_path),
        }


class Dataset:
    """Unified dataset interface for HyperGen."""
    
    def __init__(
        self,
        data: Union[ImageDataset, VideoDataset],
        dataset_type: str = "image",
    ) -> None:
        """Initialize dataset wrapper.
        
        Args:
            data: Underlying dataset
            dataset_type: Type of dataset ("image" or "video")
        """
        self.data = data
        self.dataset_type = dataset_type
        
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        dataset_type: Optional[str] = None,
        caption_file: Optional[Union[str, Path]] = None,
        metadata_file: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Dataset:
        """Load dataset from directory or file.
        
        Args:
            path: Path to dataset directory or file
            dataset_type: Type of dataset (auto-detected if None)
            caption_file: Optional caption file (JSON or TXT)
            metadata_file: Optional metadata file
            **kwargs: Additional dataset arguments
            
        Returns:
            Dataset instance
        """
        path = Path(path)
        
        if dataset_type is None:
            dataset_type = cls._detect_dataset_type(path)
            
        if dataset_type == "image":
            data = cls._load_image_dataset(path, caption_file, metadata_file, **kwargs)
        elif dataset_type == "video":
            data = cls._load_video_dataset(path, caption_file, metadata_file, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
        return cls(data, dataset_type)
        
    @staticmethod
    def _detect_dataset_type(path: Path) -> str:
        """Auto-detect dataset type from files."""
        if path.is_dir():
            image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            
            files = list(path.iterdir())[:10]
            
            for file in files:
                if file.suffix.lower() in image_exts:
                    return "image"
                elif file.suffix.lower() in video_exts:
                    return "video"
                    
        return "image"
        
    @staticmethod
    def _load_image_dataset(
        path: Path,
        caption_file: Optional[Path],
        metadata_file: Optional[Path],
        **kwargs: Any,
    ) -> ImageDataset:
        """Load image dataset from directory."""
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        
        if path.is_dir():
            image_paths = sorted([
                p for p in path.iterdir()
                if p.suffix.lower() in image_exts
            ])
        else:
            with open(path) as f:
                image_paths = [Path(line.strip()) for line in f]
                
        captions = None
        if caption_file:
            captions = Dataset._load_captions(Path(caption_file), len(image_paths))
            
        return ImageDataset(image_paths, captions, **kwargs)
        
    @staticmethod
    def _load_video_dataset(
        path: Path,
        caption_file: Optional[Path],
        metadata_file: Optional[Path],
        **kwargs: Any,
    ) -> VideoDataset:
        """Load video dataset from directory."""
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        
        if path.is_dir():
            video_paths = sorted([
                p for p in path.iterdir()
                if p.suffix.lower() in video_exts
            ])
        else:
            with open(path) as f:
                video_paths = [Path(line.strip()) for line in f]
                
        captions = None
        if caption_file:
            captions = Dataset._load_captions(Path(caption_file), len(video_paths))
            
        return VideoDataset(video_paths, captions, **kwargs)
        
    @staticmethod
    def _load_captions(caption_file: Path, num_items: int) -> List[str]:
        """Load captions from file."""
        if caption_file.suffix == ".json":
            with open(caption_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return list(data.values())
        else:
            with open(caption_file) as f:
                return [line.strip() for line in f]
                
        return [""] * num_items
        
    def create_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Create PyTorch DataLoader.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            pin_memory: Pin memory for CUDA
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            self.data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        return self.data[idx]
        
    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        """Split dataset into train/val/test sets.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            Train, validation, and test datasets
        """
        total_len = len(self.data)
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        
        indices = torch.randperm(total_len).tolist()
        
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len] if val_len > 0 else None
        test_indices = indices[train_len + val_len:] if test_ratio > 0 else None
        
        train_data = torch.utils.data.Subset(self.data, train_indices)
        val_data = torch.utils.data.Subset(self.data, val_indices) if val_indices else None
        test_data = torch.utils.data.Subset(self.data, test_indices) if test_indices else None
        
        train_dataset = Dataset(train_data, self.dataset_type)
        val_dataset = Dataset(val_data, self.dataset_type) if val_data else None
        test_dataset = Dataset(test_data, self.dataset_type) if test_data else None
        
        return train_dataset, val_dataset, test_dataset