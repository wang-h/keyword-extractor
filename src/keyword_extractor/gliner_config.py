"""
GLiNER 全局配置：预训练基座 ID 必须在「SFT 起点」与「未指定路径时的推理默认」中保持一致。

微调后的权重请用 GLiNER.from_pretrained(本地目录) 加载；该目录必须与训练时使用的基座
（本模块中的 GLINER_BASE_MODEL_ID）同属一系架构，不可与另一 Hub 模型混用。
"""
from __future__ import annotations

import os

# 可通过环境变量覆盖（CI / 实验），但训练与线上应约定为同一值
GLINER_BASE_MODEL_ID: str = os.environ.get(
    "GLINER_BASE_MODEL_ID",
    "urchade/gliner_multi-v2.1",
)

# 脚本默认写出目录；推理时若加载微调权重，通常指向此路径或 checkpoint 子目录
GLINER_DEFAULT_SFT_OUTPUT_DIR: str = os.environ.get(
    "GLINER_SFT_OUTPUT_DIR",
    "./models/gliner_sft",
)
