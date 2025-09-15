"""
AdcSR ComfyUI Custom Nodes
Adversarial Diffusion Compression for Real-World Image Super-Resolution

将 AdcSR 模型集成到 ComfyUI 中，提供图像超分辨率功能
"""

# 导入节点映射
from .adcsr_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# ComfyUI 节点注册
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# 版本信息
__version__ = "1.0.0"

print("🚀 AdcSR ComfyUI 节点已加载")
print("📦 包含节点:")
for node_key, node_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"   - {node_name}")
print("📂 节点分类: upscaling/AdcSR")
