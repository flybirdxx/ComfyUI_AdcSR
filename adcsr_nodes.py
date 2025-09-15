"""
AdcSR ComfyUI Custom Nodes Implementation
Adversarial Diffusion Compression for Real-World Image Super-Resolution

基于 AdcSR 仓库的 ComfyUI 节点实现
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入folder_paths，如果失败则使用默认路径
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False
    # folder_paths不可用，使用默认路径配置

import comfy.utils
import comfy.model_management
import numpy as np
from PIL import Image
from torchvision import transforms
import copy
import types

# 导入 AdcSR 核心模块
import sys
import os
adcsr_path = os.path.join(os.path.dirname(__file__), 'AdcSR')
sys.path.insert(0, adcsr_path)

from .AdcSR.model import Net
from .AdcSR.forward import (
    MyUNet2DConditionModel_SD_forward,
    MyCrossAttnDownBlock2D_SD_forward,
    MyDownBlock2D_SD_forward,
    MyUNetMidBlock2DCrossAttn_SD_forward,
    MyCrossAttnUpBlock2D_SD_forward,
    MyUpBlock2D_SD_forward,
    MyResnetBlock2D_SD_forward,
    MyTransformer2DModel_SD_forward
)
from .AdcSR.utils import add_lora_to_unet
from .model_downloader import AdcSRModelDownloader


class AdcSRModelLoader:
    """AdcSR Model Loader Node - 自动加载 AdcSR 模型权重"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {}
        }
    
    RETURN_TYPES = ("ADCSR_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "upscaling/AdcSR"
    
    def _auto_detect_model_paths(self):
        """完全自动检测模型文件路径 - 自动下载缺失的模型"""
        # 完全自动检测模型文件路径
        
        # 首先尝试使用folder_paths.models_dir
        if FOLDER_PATHS_AVAILABLE:
            try:
                models_dir = folder_paths.models_dir
            except:
                # 如果folder_paths不可用，使用默认路径
                current_dir = os.path.dirname(__file__)
                comfyui_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
                models_dir = os.path.join(comfyui_root, "models")
        else:
            # 如果folder_paths不可用，使用默认路径
            current_dir = os.path.dirname(__file__)
            comfyui_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
            models_dir = os.path.join(comfyui_root, "models")
            
        adcsr_dir = os.path.join(models_dir, "adcsr")
        
        # 使用folder_paths获取模型目录
        
        # 如果AdcSR目录不存在，尝试从ComfyUI根目录查找
        if not os.path.exists(adcsr_dir):
            # AdcSR目录不存在，尝试从ComfyUI根目录查找
            current_dir = os.path.dirname(__file__)  # ComfyUI/custom_nodes/ComfyUI-AdcSR
            comfyui_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))  # 回到ComfyUI-aki-v1.7
            models_dir = os.path.join(comfyui_root, "models")
            adcsr_dir = os.path.join(models_dir, "adcsr")
            
            # 从当前文件位置计算ComfyUI根目录
        
        # 如果AdcSR目录仍然不存在，创建它
        if not os.path.exists(adcsr_dir):
            # 创建AdcSR模型目录
            os.makedirs(adcsr_dir, exist_ok=True)
        
        # 初始化模型下载器
        downloader = AdcSRModelDownloader(models_dir)
        
        # 自动检测主模型文件
        model_candidates = [
            "net_params_200.pkl",
            "net_params.pkl", 
            "model.pkl",
            "adcsr_model.pkl"
        ]
        model_path = None
        for candidate in model_candidates:
            candidate_path = os.path.join(adcsr_dir, candidate)
            if os.path.exists(candidate_path):
                model_path = candidate_path
                # 找到主模型
                break
        else:
            # 未找到主模型，尝试自动下载
            print("未找到AdcSR主模型文件，尝试自动下载...")
            try:
                if downloader.check_and_download("adcsr_main"):
                    # 下载成功后，检查是否有对应的 pkl 文件
                    for candidate in model_candidates:
                        candidate_path = os.path.join(adcsr_dir, candidate)
                        if os.path.exists(candidate_path):
                            model_path = candidate_path
                            break
                    
                    if not model_path:
                        raise FileNotFoundError(f"下载完成但未找到对应的模型文件")
                else:
                    raise FileNotFoundError(f"自动下载失败，请手动将模型文件放在: {adcsr_dir}")
            except Exception as e:
                raise FileNotFoundError(f"未找到AdcSR主模型文件且自动下载失败: {e}。请将模型文件放在: {adcsr_dir}")
        
        # 自动检测半解码器文件
        decoder_candidates = [
            "halfDecoder.ckpt",
            "half_decoder.ckpt",
            "decoder.ckpt",
            "half_decoder.pth"
        ]
        half_decoder_path = None
        for candidate in decoder_candidates:
            candidate_path = os.path.join(adcsr_dir, candidate)
            if os.path.exists(candidate_path):
                half_decoder_path = candidate_path
                # 找到半解码器
                break
        else:
            # 未找到半解码器，尝试自动下载
            print("未找到半解码器文件，尝试自动下载...")
            try:
                if downloader.check_and_download("half_decoder"):
                    # 下载成功后，检查是否有对应的文件
                    for candidate in decoder_candidates:
                        candidate_path = os.path.join(adcsr_dir, candidate)
                        if os.path.exists(candidate_path):
                            half_decoder_path = candidate_path
                            break
                    
                    if not half_decoder_path:
                        raise FileNotFoundError(f"下载完成但未找到对应的半解码器文件")
                else:
                    raise FileNotFoundError(f"自动下载失败，请手动将半解码器文件放在: {adcsr_dir}")
            except Exception as e:
                raise FileNotFoundError(f"未找到半解码器文件且自动下载失败: {e}。请将文件放在: {adcsr_dir}")
        
        # 自动检测SD2.1模型
        # 首先获取diffusers文件夹路径
        if FOLDER_PATHS_AVAILABLE:
            try:
                diffusers_paths = folder_paths.get_folder_paths("diffusers")
                if diffusers_paths:
                    diffusers_dir = diffusers_paths[0]
                    # 使用folder_paths diffusers目录
                else:
                    diffusers_dir = os.path.join(models_dir, "diffusers")
                    # 使用默认diffusers目录
            except Exception as e:
                diffusers_dir = os.path.join(models_dir, "diffusers")
                # 获取diffusers路径失败，使用默认
        else:
            diffusers_dir = os.path.join(models_dir, "diffusers")
            # 使用默认diffusers目录
        
        # 检查本地SD2.1模型 - 优先检查diffusers文件夹
        sd21_local_candidates = [
            "stable-diffusion-2-1-base",
            "sd2.1-base",
            "sd21-base"
        ]
        sd21_model_path = None
        
        # 首先检查diffusers文件夹
        for candidate in sd21_local_candidates:
            candidate_path = os.path.join(diffusers_dir, candidate)
            if os.path.exists(candidate_path):
                sd21_model_path = candidate_path
                # 找到diffusers中的SD2.1模型
                break
        
        # 如果diffusers中没有，检查adcsr文件夹
        if sd21_model_path is None:
            for candidate in sd21_local_candidates:
                candidate_path = os.path.join(adcsr_dir, candidate)
                if os.path.exists(candidate_path):
                    sd21_model_path = candidate_path
                    # 找到adcsr中的SD2.1模型
                    break
        
        # 如果本地都没有，尝试自动下载
        if sd21_model_path is None:
            print("未找到SD2.1模型，尝试自动下载...")
            try:
                if downloader.check_and_download("sd21_base"):
                    # 下载成功后，检查是否有对应的目录
                    for candidate in sd21_local_candidates:
                        candidate_path = os.path.join(diffusers_dir, candidate)
                        if os.path.exists(candidate_path):
                            sd21_model_path = candidate_path
                            break
                    
                    if not sd21_model_path:
                        # 如果下载后仍然找不到，使用Hugging Face模型
                        sd21_model_path = "stabilityai/stable-diffusion-2-1-base"
                        print("下载完成但未找到本地模型，将使用Hugging Face模型")
                else:
                    # 下载失败，使用Hugging Face模型
                    sd21_model_path = "stabilityai/stable-diffusion-2-1-base"
                    print("自动下载失败，将使用Hugging Face模型")
            except Exception as e:
                # 下载出错，使用Hugging Face模型
                sd21_model_path = "stabilityai/stable-diffusion-2-1-base"
                print(f"自动下载出错: {e}，将使用Hugging Face模型")
        
        # 验证文件存在性
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"主模型文件不存在: {model_path}")
        if not os.path.exists(half_decoder_path):
            raise FileNotFoundError(f"半解码器文件不存在: {half_decoder_path}")
        
        # 模型路径检测完成
        return {
            'model_path': model_path,
            'half_decoder_path': half_decoder_path,
            'sd21_model_path': sd21_model_path
        }

    def load_model(self):
        """自动加载 AdcSR 模型 - 完全自动检测，无需用户输入"""
        try:
            device = comfy.model_management.get_torch_device()
            # 开始自动加载 AdcSR 模型
            
            # 完全自动检测模型路径
            model_paths = self._auto_detect_model_paths()
            model_path = model_paths['model_path']
            half_decoder_path = model_paths['half_decoder_path']
            sd21_model_path = model_paths['sd21_model_path']
            
            # 获取模型路径
            
            # Step 1: 加载 Stable Diffusion 2.1 基础模型
            # 加载 Stable Diffusion 2.1 基础模型
            from diffusers import StableDiffusionPipeline
            
            # 检查是否为本地路径
            if os.path.exists(sd21_model_path):
                # 使用本地模型路径
                pipe = StableDiffusionPipeline.from_pretrained(sd21_model_path).to(device)
            else:
                # 使用 Hugging Face 模型
                pipe = StableDiffusionPipeline.from_pretrained(sd21_model_path).to(device)
            
            vae = pipe.vae
            unet = pipe.unet
            
            # Step 2: 加载预训练的 halfDecoder
            # 加载预训练解码器
            ckpt_halfdecoder = torch.load(half_decoder_path, weights_only=False)
            
            from diffusers.models.autoencoders.vae import Decoder
            decoder = Decoder(
                in_channels=4,
                out_channels=3,
                up_block_types=["UpDecoderBlock2D" for _ in range(4)],
                block_out_channels=[64, 128, 256, 256],
                layers_per_block=2, 
                norm_num_groups=32, 
                act_fn="silu", 
                norm_type="group", 
                mid_block_add_attention=True
            ).to(device)
            
            # 加载解码器权重
            decoder_ckpt = {}
            for k, v in ckpt_halfdecoder["state_dict"].items():
                if "decoder" in k:
                    new_k = k.replace("decoder.", "")
                    decoder_ckpt[new_k] = v
            decoder.load_state_dict(decoder_ckpt, strict=True)
            
            # Step 3: 创建 Net 模型
            # 创建 AdcSR Net 模型
            # 移除DataParallel包装以减少内存使用
            model = Net(unet, copy.deepcopy(decoder)).to(device)
            
            # Step 4: 加载 AdcSR 核心权重
            # 加载 AdcSR 核心权重
            state_dict = torch.load(model_path, weights_only=False)
            
            # 处理DataParallel保存的权重（移除module.前缀）
            if any(key.startswith('module.') for key in state_dict.keys()):
                # 检测到DataParallel权重，移除module.前缀
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('module.'):
                        new_key = key[7:]  # 移除 'module.' 前缀
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            model.load_state_dict(state_dict, strict=True)
            
            # Step 5: 构建最终推理模型
            # 构建最终推理模型
            final_model = torch.nn.Sequential(
                model,
                *decoder.up_blocks,
                decoder.conv_norm_out,
                decoder.conv_act,
                decoder.conv_out,
            ).to(device)
            
            # 创建 AdcSR 模型对象
            adcsr_model = {
                'model': final_model,
                'device': device,
                'model_path': model_path,
                'half_decoder_path': half_decoder_path
            }
            
            # AdcSR 模型加载完成
            return (adcsr_model,)
            
        except Exception as e:
            print(f"❌ 加载 AdcSR 模型时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e


class AdcSRUpscaler:
    """AdcSR Upscaler Node - 执行超分辨率推理"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adcsr_model": ("ADCSR_MODEL",),
                "image": ("IMAGE",),
                "tile_size": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 512,
                    "step": 32,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "upscaling/AdcSR"

    def upscale(self, adcsr_model, image, tile_size=256):
        """执行 AdcSR 超分辨率推理"""
        try:
            # 开始 AdcSR 超分辨率处理
            
            device = adcsr_model['device']
            model = adcsr_model['model']
            
            # Step 1: 图像预处理
            # 预处理输入图像
            
            # 确保图像是正确的格式 [B, H, W, C]
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # 添加批次维度 [1, H, W, C]
            
            # 转换为 PIL 格式进行预处理
            batch_size = image.shape[0]
            processed_images = []
            
            for i in range(batch_size):
                # 转换为 PIL Image
                img_tensor = image[i]  # [H, W, C]
                img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_array)
                
                # 应用 AdcSR 预处理: ToTensor() * 2 - 1
                img_tensor = transforms.ToTensor()(pil_image).to(device).unsqueeze(0) * 2 - 1
                processed_images.append(img_tensor)
            
            # 合并批次
            if len(processed_images) == 1:
                lr_tensor = processed_images[0]
            else:
                lr_tensor = torch.cat(processed_images, dim=0)
            
            # 预处理完成
            
            # Step 1.5: 检查并修复尺寸问题
            # 检查输入尺寸
            original_shape = lr_tensor.shape
            batch_size, channels, height, width = original_shape
            
            # 计算需要填充的像素数 - 使用64的倍数填充策略
            # 确保所有下采样层的尺寸都是8的倍数，避免跳跃连接中的尺寸不匹配
            pad_h = (64 - height % 64) % 64
            pad_w = (64 - width % 64) % 64
            
            total_pad_h = pad_h
            total_pad_w = pad_w
            
            if total_pad_h > 0 or total_pad_w > 0:
                # 需要填充以匹配模型要求
                
                # 使用反射填充
                lr_tensor = torch.nn.functional.pad(lr_tensor, (0, total_pad_w, 0, total_pad_h), mode='reflect')
                # 填充完成
            
            # Step 2: 执行推理 - 使用分块处理避免OOM
            # 执行 AdcSR 推理
            # 设置模型为评估模式并清理GPU缓存
            model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                # 检查是否需要分块处理
                height, width = lr_tensor.shape[2], lr_tensor.shape[3]
                
                if height > tile_size or width > tile_size:
                    # 图像尺寸较大，使用分块处理
                    sr_tensor = self._tiled_inference(model, lr_tensor, tile_size)
                else:
                    # 图像尺寸较小，直接处理
                    sr_tensor = model(lr_tensor)
            
            # 推理完成
            
            # Step 3: 裁剪掉填充的部分，保持超分辨率尺寸
            if total_pad_h > 0 or total_pad_w > 0:
                # 计算超分辨率后的原始尺寸（4倍）
                sr_original_h = original_shape[2] * 4  # 4倍超分辨率
                sr_original_w = original_shape[3] * 4
                # 裁剪掉填充部分，保持超分辨率尺寸
                # 裁剪掉填充的部分，但保持4倍超分辨率
                sr_tensor = sr_tensor[:, :, :sr_original_h, :sr_original_w]
                # 裁剪完成
            
            # Step 4: 后处理 - 色彩校正
            # 应用色彩校正
            # 使用原始未填充的图像进行色彩校正（与原始AdcSR代码一致）
            lr_original = lr_tensor[:, :, :original_shape[2], :original_shape[3]] if total_pad_h > 0 or total_pad_w > 0 else lr_tensor
            # 直接使用原始LR图像进行色彩校正，不进行上采样
            # SR = (SR - SR.mean) / SR.std * LR.std + LR.mean
            sr_tensor = (sr_tensor - sr_tensor.mean(dim=[2,3], keepdim=True)) / sr_tensor.std(dim=[2,3], keepdim=True) \
                       * lr_original.std(dim=[2,3], keepdim=True) + lr_original.mean(dim=[2,3], keepdim=True)
            
            # Step 5: 转换回 [0, 1] 范围并转换为 ComfyUI 格式
            # 转换输出格式
            # 从 [-1, 1] 转换到 [0, 1]
            sr_tensor = (sr_tensor / 2 + 0.5).clamp(0, 1)
            
            # 转换为 ComfyUI 格式 [B, H, W, C]
            if sr_tensor.shape[1] == 3:  # [B, C, H, W] -> [B, H, W, C]
                sr_tensor = sr_tensor.permute(0, 2, 3, 1)
            
            # 输出格式转换完成
            
            # AdcSR 超分辨率处理完成
            return (sr_tensor,)
            
        except Exception as e:
            print(f"❌ AdcSR 超分辨率处理出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _tiled_inference(self, model, lr_tensor, tile_size=256, overlap=32):
        """
        分块推理方法，避免大图像OOM问题
        
        Args:
            model: AdcSR模型
            lr_tensor: 输入张量 [B, C, H, W]
            tile_size: 分块尺寸
            overlap: 重叠像素数，用于避免边界伪影
        
        Returns:
            sr_tensor: 超分辨率结果 [B, C, H*4, W*4]
        """
        batch_size, channels, height, width = lr_tensor.shape
        device = lr_tensor.device
        
        # 计算输出尺寸
        sr_height = height * 4
        sr_width = width * 4
        
        # 初始化输出张量
        sr_tensor = torch.zeros(batch_size, channels, sr_height, sr_width, device=device, dtype=lr_tensor.dtype)
        
        # 计算分块数量
        tiles_h = (height + tile_size - 1) // tile_size
        tiles_w = (width + tile_size - 1) // tile_size
        
        # 分块处理
        
        # 处理每个分块
        for i in range(tiles_h):
            for j in range(tiles_w):
                # 计算分块位置
                start_h = i * tile_size
                start_w = j * tile_size
                end_h = min(start_h + tile_size, height)
                end_w = min(start_w + tile_size, width)
                
                # 添加重叠区域
                pad_start_h = max(0, start_h - overlap)
                pad_start_w = max(0, start_w - overlap)
                pad_end_h = min(height, end_h + overlap)
                pad_end_w = min(width, end_w + overlap)
                
                # 提取分块
                tile = lr_tensor[:, :, pad_start_h:pad_end_h, pad_start_w:pad_end_w]
                
                # 对分块进行推理
                with torch.no_grad():
                    sr_tile = model(tile)
                
                # 计算在输出中的位置
                sr_start_h = start_h * 4
                sr_start_w = start_w * 4
                sr_end_h = end_h * 4
                sr_end_w = end_w * 4
                
                # 计算重叠区域在输出中的位置
                sr_pad_start_h = pad_start_h * 4
                sr_pad_start_w = pad_start_w * 4
                sr_pad_end_h = pad_end_h * 4
                sr_pad_end_w = pad_end_w * 4
                
                # 裁剪掉重叠区域
                crop_start_h = (start_h - pad_start_h) * 4
                crop_start_w = (start_w - pad_start_w) * 4
                crop_end_h = crop_start_h + (end_h - start_h) * 4
                crop_end_w = crop_start_w + (end_w - start_w) * 4
                
                sr_tile_cropped = sr_tile[:, :, crop_start_h:crop_end_h, crop_start_w:crop_end_w]
                
                # 将结果写入输出张量
                sr_tensor[:, :, sr_start_h:sr_end_h, sr_start_w:sr_end_w] = sr_tile_cropped
                
                # 清理内存
                del tile, sr_tile, sr_tile_cropped
                torch.cuda.empty_cache()
        
        # 分块处理完成
        return sr_tensor




# 节点注册
NODE_CLASS_MAPPINGS = {
    "AdcSRModelLoader": AdcSRModelLoader,
    "AdcSRUpscaler": AdcSRUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdcSRModelLoader": "AdcSR Model Loader",
    "AdcSRUpscaler": "AdcSR Upscaler",
}
