"""
GLiNER 训练与推理使用的统一标签集合（多类型实体 / 关键词）
"""
from typing import Dict, List

# 与 prepare_training_data.map_tag_type_to_label 一致
GLINER_TRAINING_LABELS: List[str] = [
    "科技公司全称",
    "软件产品全称",
    "AI模型及版本号",
    "核心技术术语",
    "硬件设备名称",
    "知名人名",
    "技术实体",
]

# 推理时按标签类型微调阈值（Bi-Encoder / 后验过滤）
def map_tag_type_to_label(tag_type: str) -> str:
    """tags.csv type / LLM 输出 type -> GLiNER 标签名。"""
    type_map = {
        "company": "科技公司全称",
        "product": "软件产品全称",
        "ai_model": "AI模型及版本号",
        "technology": "核心技术术语",
        "hardware": "硬件设备名称",
        "person": "知名人名",
    }
    return type_map.get((tag_type or "").lower().strip(), "技术实体")


DEFAULT_LABEL_THRESHOLDS: Dict[str, float] = {
    "科技公司全称": 0.42,
    "软件产品全称": 0.38,
    "AI模型及版本号": 0.4,
    "核心技术术语": 0.35,
    "硬件设备名称": 0.4,
    "知名人名": 0.45,
    "技术实体": 0.33,
}
