<div align="center">

<a href="https://hypergen.ai"><picture>

<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/hypergen/hypergen/main/assets/hypergen_logo_white.png">
<source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/hypergen/hypergen/main/assets/hypergen_logo_black.png">
<img alt="hypergen logo" src="https://raw.githubusercontent.com/hypergen/hypergen/main/assets/hypergen_logo_black.png" height="110" style="max-width: 100%;">
</picture></a>

<a href="https://colab.research.google.com/github/hypergen/notebooks/blob/main/quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" width="150"></a>
<a href="https://discord.gg/hypergen"><img src="https://img.shields.io/discord/1234567890?color=5865F2&logo=discord&logoColor=white&label=Discord" width="150"></a>
<a href="https://docs.hypergen.ai"><img src="https://img.shields.io/badge/docs-hypergen.ai-blue" width="120"></a>

### Train & run image and video diffusion models 3x faster with 80% less VRAM!

![Performance Chart](https://i.ibb.co/performance-chart.png)

</div>

## ✨ Simple as 5 Lines

HyperGen makes training and running diffusion models incredibly simple. No lengthy notebooks, no complex configurations - just clean Python code:

```python
import hypergen

model = hypergen.Model.load("flux/flux-dev")
dataset = hypergen.Dataset.load("/path/to/images")
lora = model.train_lora(dataset)
image = model.run_lora(prompt="A cat holding a sign that says hello world", lora=lora)
```

That's it! HyperGen handles optimization, memory management, and acceleration automatically.

## 🚀 Supported Models

| Model Family           | Type  | Memory Reduction | Speed Improvement | Colab Notebook                                                                                               |
| ---------------------- | ----- | ---------------- | ----------------- | ------------------------------------------------------------------------------------------------------------ |
| **FLUX.1**             | Image | 80% less VRAM    | 3x faster         | [▶️ Try now](https://colab.research.google.com/github/hypergen/notebooks/blob/main/flux_training.ipynb)      |
| **Stable Diffusion 3** | Image | 75% less VRAM    | 2.8x faster       | [▶️ Try now](https://colab.research.google.com/github/hypergen/notebooks/blob/main/sd3_training.ipynb)       |
| **SDXL**               | Image | 70% less VRAM    | 2.5x faster       | [▶️ Try now](https://colab.research.google.com/github/hypergen/notebooks/blob/main/sdxl_training.ipynb)      |
| **CogVideoX**          | Video | 85% less VRAM    | 3.2x faster       | [▶️ Try now](https://colab.research.google.com/github/hypergen/notebooks/blob/main/cogvideox_training.ipynb) |
| **Luma Dream Machine** | Video | 80% less VRAM    | 3x faster         | [▶️ Try now](https://colab.research.google.com/github/hypergen/notebooks/blob/main/luma_training.ipynb)      |
| **Sora (Replica)**     | Video | 82% less VRAM    | 2.9x faster       | [▶️ Try now](https://colab.research.google.com/github/hypergen/notebooks/blob/main/sora_training.ipynb)      |

- See [all supported models](https://docs.hypergen.ai/models) and [performance benchmarks](https://docs.hypergen.ai/benchmarks)
- Browse our [model zoo](https://huggingface.co/hypergen) on Hugging Face
- Check out [community fine-tunes](https://docs.hypergen.ai/community)

## ⚡ Installation

### Quick Install

```bash
pip install hypergen
```

### From Source

```bash
git clone https://github.com/hypergen/hypergen.git
cd hypergen
pip install -e .
```

### Docker

```bash
docker run -it --gpus all hypergen/hypergen:latest
```

## ☄️ Why HyperGen?

**🎯 Dead Simple API**: 5 lines vs 500+ line notebooks. Focus on your ideas, not infrastructure.

**⚡ Blazing Fast**: 3x faster training and inference with 80% less VRAM usage by default.

**🔧 Zero Configuration**: Automatic optimization detection and memory management.

**🌐 Universal**: Works with image and video models from any provider.

**🛡️ Production Ready**: Used by companies generating millions of images daily.

## 📖 Quick Examples

### Image Generation

```python
import hypergen

# Load any diffusion model
model = hypergen.Model.load("runwayml/stable-diffusion-v1-5")

# Generate images
images = model.generate([
    "A serene mountain landscape at sunset",
    "A cyberpunk cityscape with neon lights"
], batch_size=2)

# Save results
hypergen.save_images(images, "outputs/")
```

### LoRA Training

```python
import hypergen

# Load model and dataset
model = hypergen.Model.load("black-forest-labs/FLUX.1-dev")
dataset = hypergen.Dataset.load("./my_photos", format="folder")

# Train LoRA with automatic optimization
lora = model.train_lora(
    dataset,
    steps=1000,
    learning_rate="auto",  # Automatic learning rate scheduling
    batch_size="auto"      # Automatic batch size optimization
)

# Use the trained LoRA
image = model.run_lora(
    prompt="A professional headshot in the style of my photos",
    lora=lora,
    strength=0.8
)
```

### Video Generation

```python
import hypergen

# Load video model
model = hypergen.Model.load("THUDM/CogVideoX-5b")

# Generate video
video = model.generate_video(
    prompt="A golden retriever playing in a sunlit meadow",
    duration=5.0,  # seconds
    fps=24
)

hypergen.save_video(video, "golden_retriever.mp4")
```

### Batch Processing

```python
import hypergen

model = hypergen.Model.load("flux/flux-dev")
prompts = hypergen.Dataset.load("prompts.txt")

# Process thousands of prompts efficiently
for batch in prompts.batch(32):
    images = model.generate(batch.prompts)
    hypergen.save_images(images, f"batch_{batch.id}/")
```

## 🚀 Performance Benchmarks

We tested HyperGen against standard implementations across different hardware configurations:

### FLUX.1 Training (LoRA, 1000 steps)

| Hardware  | ☄️ HyperGen | Standard  | Memory   | Speed       |
| --------- | ----------- | --------- | -------- | ----------- |
| RTX 4090  | 8GB VRAM    | 22GB VRAM | 80% less | 3.2x faster |
| A100 40GB | 12GB VRAM   | 38GB VRAM | 75% less | 2.8x faster |
| A100 80GB | 18GB VRAM   | 76GB VRAM | 80% less | 3.1x faster |

### CogVideoX Generation (16 frames, 720p)

| Hardware  | ☄️ HyperGen | Standard  | Memory   | Speed       |
| --------- | ----------- | --------- | -------- | ----------- |
| RTX 4090  | 14GB VRAM   | OOM       | 85% less | 3x faster   |
| A100 40GB | 22GB VRAM   | 38GB VRAM | 82% less | 3.3x faster |
| A100 80GB | 28GB VRAM   | 72GB VRAM | 85% less | 3.1x faster |

_Benchmarks conducted with fp16 precision, batch size optimized for each setup_

## 🔧 Advanced Features

### Custom Optimization

```python
import hypergen

model = hypergen.Model.load("flux/flux-dev")
model.configure(
    precision="bf16",           # or fp16, fp32
    attention_backend="flash",  # flash, xformers, native
    memory_efficient=True,      # Enable gradient checkpointing
    compile_model=True         # PyTorch 2.0 compilation
)
```

### Multi-GPU Training

```python
import hypergen

model = hypergen.Model.load("flux/flux-dev", num_gpus=4)
dataset = hypergen.Dataset.load("./large_dataset")

lora = model.train_lora(
    dataset,
    strategy="ddp",  # or fsdp, deepspeed
    steps=5000
)
```

### Custom Datasets

```python
import hypergen

# From Hugging Face
dataset = hypergen.Dataset.load("username/my-dataset")

# From local folder
dataset = hypergen.Dataset.load("./images", format="folder")

# From URLs
dataset = hypergen.Dataset.load([
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
])

# Custom preprocessing
dataset = dataset.preprocess(
    resize=(512, 512),
    crop="center",
    normalize=True
)
```

## 📚 Documentation

- 📖 [Getting Started Guide](https://docs.hypergen.ai/getting-started)
- 🎯 [API Reference](https://docs.hypergen.ai/api)
- 🏗️ [Architecture Overview](https://docs.hypergen.ai/architecture)
- 🔧 [Advanced Usage](https://docs.hypergen.ai/advanced)
- 🤝 [Contributing](https://docs.hypergen.ai/contributing)
- 🐛 [Troubleshooting](https://docs.hypergen.ai/troubleshooting)

## 🌟 Key Optimizations

**Memory Optimizations**:

- Gradient checkpointing with smart activation recomputation
- Dynamic attention scaling and memory-efficient cross-attention
- Automatic mixed precision with loss scaling
- Smart caching and memory defragmentation

**Speed Optimizations**:

- Custom CUDA kernels for common operations
- PyTorch 2.0 compilation with dynamic shapes
- Optimized attention mechanisms (Flash Attention, xFormers)
- Automatic batch size and learning rate scheduling

**Training Optimizations**:

- LoRA with rank adaptation and smart target module selection
- Gradient accumulation with automatic scaling
- Advanced sampling strategies and data loading
- Multi-GPU training with optimal communication patterns

## 🤝 Community & Support

| Platform             | Link                                                       | Description                       |
| -------------------- | ---------------------------------------------------------- | --------------------------------- |
| 📚 **Documentation** | [docs.hypergen.ai](https://docs.hypergen.ai)               | Complete guides and API reference |
| 💬 **Discord**       | [Join our Discord](https://discord.gg/hypergen)            | Community support and discussions |
| 🐙 **GitHub Issues** | [Report bugs](https://github.com/hypergen/hypergen/issues) | Bug reports and feature requests  |
| 🐦 **Twitter**       | [@hypergen](https://twitter.com/hypergen)                  | Updates and announcements         |
| 📧 **Email**         | support@hypergen.ai                                        | Enterprise support                |

## 🔄 Migration from Other Frameworks

### From Diffusers

```python
# Before (diffusers)
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipe("A cat").images[0]

# After (hypergen)
import hypergen
model = hypergen.Model.load("runwayml/stable-diffusion-v1-5")
image = model.generate("A cat")
```

### From Training Scripts

```python
# Before (100+ lines of training code)
# ... complex setup, data loading, training loops ...

# After (hypergen)
import hypergen
model = hypergen.Model.load("flux/flux-dev")
dataset = hypergen.Dataset.load("./data")
lora = model.train_lora(dataset, steps=1000)
```

## 🏆 Showcase

Models trained with HyperGen:

- [HyperGen-FLUX-Portraits](https://huggingface.co/hypergen-models/flux-portraits) - Professional portrait LoRA
- [HyperGen-CogVideoX-Nature](https://huggingface.co/hypergen-models/cogvideox-nature) - Nature documentary style
- [HyperGen-SDXL-Architecture](https://huggingface.co/hypergen-models/sdxl-architecture) - Architectural visualization

_Want to showcase your HyperGen model? [Submit here](https://docs.hypergen.ai/showcase)_

## 📄 License

HyperGen is released under the [Apache 2.0 License](LICENSE).

## 🙏 Acknowledgments

HyperGen builds upon the incredible work of:

- [🤗 Hugging Face Diffusers](https://github.com/huggingface/diffusers) - Core diffusion model implementations
- [PyTorch](https://pytorch.org) - Deep learning framework
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Efficient attention mechanisms
- [xFormers](https://github.com/facebookresearch/xformers) - Memory-efficient transformers
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning

Special thanks to our [contributors](https://github.com/hypergen/hypergen/graphs/contributors) and the open-source AI community.

---

<div align="center">

**Built with ❤️ by the HyperGen team**

[Website](https://hypergen.ai) • [Documentation](https://docs.hypergen.ai) • [Discord](https://discord.gg/hypergen) • [Twitter](https://twitter.com/hypergen_ai)

_If HyperGen accelerated your diffusion models, please ⭐ this repo and [share your results](https://docs.hypergen.ai/showcase)!_

</div>
