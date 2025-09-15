# AdcSR ComfyUI Node

A ComfyUI custom node implementation based on [AdcSR (Adversarial Diffusion Compression for Real-World Image Super-Resolution)](https://huggingface.co/Guaishou74851/AdcSR).

## Download & Installation

### Method 1: Git Clone (Recommended)
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/flybirdxx/ComfyUI_AdcSR.git
```

### Method 2: Manual Download
1. Visit the [project page](https://github.com/flybirdxx/ComfyUI_AdcSR)
2. Click "Code" → "Download ZIP"
3. Extract to `ComfyUI/custom_nodes/` directory
4. Rename the folder to `ComfyUI_AdcSR`

## Features

- **Automatic Model Detection**: Automatically detects local model files
- **Smart Download**: Automatically downloads missing models from Hugging Face
- **Efficient Processing**: Supports image super-resolution processing
- **Memory Optimization**: Memory optimization for large images

## Required Models to Download

```
models/
    ├── adcsr/                          # AdcSR model directory
    │   ├── net_params_200.pkl         # AdcSR main model (1.7GB)
    │   └── halfDecoder.ckpt           # AdcSR half decoder (360MB)
    └── diffusers/                     # Diffusers model directory
        └── stable-diffusion-2-1-base/ # SD2.1 Base model (5GB)
            ├── model_index.json
            ├── unet/
            │   ├── config.json
            │   └── diffusion_pytorch_model.safetensors
            ├── vae/
            │   ├── config.json
            │   └── diffusion_pytorch_model.safetensors
            ├── text_encoder/
            │   ├── config.json
            │   └── model.safetensors
            ├── tokenizer/
            │   ├── tokenizer.json
            │   ├── tokenizer_config.json
            │   └── vocab.json
            └── scheduler/
                └── scheduler_config.json
```

## Automatic Download

When using the node for the first time, the system will automatically detect missing models and download them:

1. **AdcSR Model**: Downloads to `ComfyUI/models/adcsr/`
2. **SD2.1 Model**: Downloads to `ComfyUI/models/diffusers/stable-diffusion-2-1-base/`

## Usage

### 1. AdcSR Model Loader

Automatically loads the AdcSR model without manual configuration.

**Input**: None
**Output**: `ADCSR_MODEL` - Loaded model object

### 2. AdcSR Upscaler

Performs super-resolution processing on images.

**Input**:
- `image`: Input image
- `adcsr_model`: AdcSR model object
- `scale_factor`: Scale factor (2 or 4)
- `tile_size`: Tile size (default: 640)
- `overlap`: Overlap pixels (default: 32)

**Output**:
- `image`: Super-resolution processed image

## Installation Requirements

```bash
pip install torch torchvision diffusers transformers huggingface_hub tqdm
```

## Important Notes

1. **First Use**: Required model files will be automatically downloaded on first run
2. **Network Requirements**: Stable internet connection required for model downloads
3. **Storage Space**: Ensure sufficient disk space for model files (approximately 7GB total)
4. **Memory Usage**: Pay attention to memory usage when processing large images
5. **Model Locations**: 
   - AdcSR models saved in `models/adcsr/` directory
   - SD2.1 models saved in `models/diffusers/` directory

## Technical Details

- Implementation based on AdcSR paper
- Supports 2x and 4x super-resolution
- Uses tiling processing to avoid memory overflow
- Integrates Stable Diffusion 2.1 as base model

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{chen2025adversarial,
  title={Adversarial Diffusion Compression for Real-World Image Super-Resolution},
  author={Chen, Bin and Li, Gehui and Wu, Rongyuan and Zhang, Xindong and Chen, Jie and Zhang, Jian and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on the [GitHub repository](https://github.com/flybirdxx/ComfyUI_AdcSR).
