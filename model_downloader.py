"""
AdcSR Model Downloader
优化的模型下载器，只下载必要的模型文件
"""

import os
import requests
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm
import json

class AdcSRModelDownloader:
    """AdcSR 模型下载器 - 只下载必要的模型文件"""
    
    def __init__(self, models_dir=None):
        """初始化下载器"""
        if models_dir is None:
            # 自动检测模型目录
            current_dir = os.path.dirname(__file__)
            # 从 ComfyUI/custom_nodes/ComfyUI-AdcSR 回到 ComfyUI 目录
            comfyui_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
            models_dir = os.path.join(comfyui_root, "models")
        
        self.models_dir = models_dir
        self.adcsr_dir = os.path.join(models_dir, "adcsr")
        self.diffusers_dir = os.path.join(models_dir, "diffusers")
        
        # 确保目录存在
        os.makedirs(self.adcsr_dir, exist_ok=True)
        os.makedirs(self.diffusers_dir, exist_ok=True)
        
        # 模型配置 - 只包含必要的文件
        self.model_configs = {
            "adcsr_main": {
                "description": "AdcSR 主模型",
                "filename": "net_params_200.pkl",
                "repo_id": "Guaishou74851/AdcSR",
                "file_path": "net_params_200.pkl",
                "size": "1.7GB",
                "local_path": os.path.join(self.adcsr_dir, "net_params_200.pkl")
            },
            "half_decoder": {
                "description": "AdcSR 半解码器",
                "filename": "halfDecoder.ckpt", 
                "repo_id": "Guaishou74851/AdcSR",
                "file_path": "halfDecoder.ckpt",
                "size": "360MB",
                "local_path": os.path.join(self.adcsr_dir, "halfDecoder.ckpt")
            },
            "sd21_base": {
                "description": "Stable Diffusion 2.1 Base",
                "filename": "stable-diffusion-2-1-base",
                "repo_id": "stabilityai/stable-diffusion-2-1-base",
                "file_path": None,  # 整个仓库
                "size": "5GB",
                "local_path": os.path.join(self.diffusers_dir, "stable-diffusion-2-1-base")
            }
        }
    
    def check_model_exists(self, model_type):
        """检查模型文件是否存在"""
        config = self.model_configs.get(model_type)
        if not config:
            return False
        
        local_path = config["local_path"]
        
        if model_type == "sd21_base":
            # 检查SD2.1模型目录和关键文件
            return (os.path.exists(local_path) and 
                   os.path.exists(os.path.join(local_path, "model_index.json")) and
                   os.path.exists(os.path.join(local_path, "unet", "config.json")))
        else:
            # 检查单个文件
            return os.path.exists(local_path)
    
    def download_file_with_progress(self, url, local_path, description="Downloading"):
        """带进度条的文件下载"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    def download_huggingface_file(self, repo_id, file_path, local_path, description="Downloading"):
        """从Hugging Face下载单个文件"""
        try:
            print(f"正在从 {repo_id} 下载 {file_path}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                local_dir=os.path.dirname(local_path),
                local_dir_use_symlinks=False
            )
            
            # 如果下载到不同位置，移动到目标位置
            if downloaded_path != local_path:
                os.rename(downloaded_path, local_path)
            
            print(f"✅ {description} 下载完成: {local_path}")
            return True
            
        except Exception as e:
            print(f"❌ {description} 下载失败: {str(e)}")
            return False
    
    def download_huggingface_repo(self, repo_id, local_path, description="Downloading"):
        """从Hugging Face下载整个仓库"""
        try:
            print(f"正在从 {repo_id} 下载整个仓库...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_path,
                local_dir_use_symlinks=False
            )
            print(f"✅ {description} 下载完成: {local_path}")
            return True
            
        except Exception as e:
            print(f"❌ {description} 下载失败: {str(e)}")
            return False
    
    def download_model(self, model_type):
        """下载指定模型"""
        if model_type not in self.model_configs:
            print(f"❌ 未知的模型类型: {model_type}")
            return False
        
        config = self.model_configs[model_type]
        
        # 检查是否已存在
        if self.check_model_exists(model_type):
            print(f"✅ {config['description']} 已存在，跳过下载")
            return True
        
        print(f"🔄 开始下载 {config['description']} ({config['size']})...")
        
        if model_type == "sd21_base":
            # 下载整个SD2.1仓库
            return self.download_huggingface_repo(
                config["repo_id"], 
                config["local_path"],
                config["description"]
            )
        else:
            # 下载单个文件
            return self.download_huggingface_file(
                config["repo_id"],
                config["file_path"], 
                config["local_path"],
                config["description"]
            )
    
    def check_and_download(self, model_type):
        """检查并下载模型（如果不存在）"""
        if self.check_model_exists(model_type):
            return True
        
        return self.download_model(model_type)
    
    def download_essential_models(self):
        """下载所有必要的模型"""
        print("🔄 开始检查并下载必要的模型文件...")
        
        success_count = 0
        total_count = len(self.model_configs)
        
        for model_type in self.model_configs.keys():
            if self.check_and_download(model_type):
                success_count += 1
            else:
                print(f"⚠️ {model_type} 下载失败")
        
        print(f"\n📊 下载完成: {success_count}/{total_count} 个模型成功")
        return success_count == total_count
    
    def get_model_info(self, model_type):
        """获取模型信息"""
        config = self.model_configs.get(model_type)
        if not config:
            return None
        
        return {
            "description": config["description"],
            "filename": config["filename"],
            "size": config["size"],
            "local_path": config["local_path"],
            "exists": self.check_model_exists(model_type)
        }
    
    def list_models(self):
        """列出所有模型状态"""
        print("\n📋 模型文件状态:")
        print("-" * 60)
        
        for model_type, config in self.model_configs.items():
            exists = self.check_model_exists(model_type)
            status = "✅ 已存在" if exists else "❌ 缺失"
            print(f"{config['description']:<25} | {config['size']:<8} | {status}")
        
        print("-" * 60)
