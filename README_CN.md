# AdcSR ComfyUI 节点

基于 [AdcSR (Adversarial Diffusion Compression for Real-World Image Super-Resolution)](https://huggingface.co/Guaishou74851/AdcSR) 的 ComfyUI 自定义节点实现。

```
当前实现对分辨率小的图片有奇效,在我本地5060TI的机器上处理超过640的图片就会爆显存,使用分块又会有很明显的接缝和色块,暂时只能这样了,因为我本人不懂代码,没有能力优化,如果有哪位大佬感兴趣,欢迎PR
```

## 下载安装

### 方法一：Git 克隆（推荐）
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/flybirdxx/ComfyUI_AdcSR.git
```

### 方法二：手动下载
1. 访问 [项目页面](https://github.com/flybirdxx/ComfyUI_AdcSR)
2. 点击 "Code" → "Download ZIP"
3. 解压到 `ComfyUI/custom_nodes/` 目录
4. 重命名文件夹为 `ComfyUI_AdcSR`

## 功能特点

- **自动模型检测**: 自动检测本地模型文件
- **智能下载**: 缺失模型时自动从 Hugging Face 下载
- **高效处理**: 支持图像超分辨率处理
- **内存优化**: 针对大图像进行内存优化

## 效果对比演示


![AdcSR 超分辨率效果](assets/20250915_181243.gif)

*上图展示了 AdcSR 模型在图像超分辨率处理中的效果对比，可以看到 AdcSR 相比传统方法在细节保持和图像质量方面有显著提升。*


## 所需下载的模型

```
models/
    ├── adcsr/                          # AdcSR 模型目录
    │   ├── net_params_200.pkl         # AdcSR 主模型 (1.7GB)
    │   └── halfDecoder.ckpt           # AdcSR 半解码器 (360MB)
    └── diffusers/                     # Diffusers 模型目录
        └── stable-diffusion-2-1-base/ # SD2.1 Base 模型 (5GB)
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

## 自动下载

当首次使用节点时，系统会自动检测缺失的模型并下载：

1. **AdcSR 模型**: 下载到 `ComfyUI/models/adcsr/`
2. **SD2.1 模型**: 下载到 `ComfyUI/models/diffusers/stable-diffusion-2-1-base/`

## 使用方法

### 1. AdcSR Model Loader

自动加载 AdcSR 模型，无需手动配置。

**输入**: 无
**输出**: `ADCSR_MODEL` - 加载的模型对象

### 2. AdcSR Upscaler

对图像进行超分辨率处理。

**输入**:
- `image`: 输入图像
- `adcsr_model`: AdcSR 模型对象
- `scale_factor`: 缩放因子 (2 或 4)
- `tile_size`: 分块大小 (默认: 640)
- `overlap`: 重叠像素 (默认: 32)

**输出**:
- `image`: 超分辨率处理后的图像

## 安装要求

```bash
pip install torch torchvision diffusers transformers huggingface_hub tqdm
```

## 注意事项

1. **首次使用**: 首次运行时会自动下载必要的模型文件
2. **网络要求**: 需要稳定的网络连接以下载模型
3. **存储空间**: 确保有足够的磁盘空间存储模型文件 (总计约 7GB)
4. **内存使用**: 处理大图像时注意内存使用情况
5. **模型位置**: 
   - AdcSR 模型保存在 `models/adcsr/` 目录
   - SD2.1 模型保存在 `models/diffusers/` 目录

## 技术细节

- 基于 AdcSR 论文实现
- 支持 2x 和 4x 超分辨率
- 使用分块处理避免内存溢出
- 集成 Stable Diffusion 2.1 作为基础模型

## 引用

如果您使用了此实现，请引用原始论文：

```bibtex
@inproceedings{chen2025adversarial,
  title={Adversarial Diffusion Compression for Real-World Image Super-Resolution},
  author={Chen, Bin and Li, Gehui and Wu, Rongyuan and Zhang, Xindong and Chen, Jie and Zhang, Jian and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
