"""
实体提取后处理规则

问题：GLiNER 提取的实体不完整或有噪音
解决：规则-based 修正和合并
"""
import re
from typing import List, Dict

# 实体别名映射（标准化）
ENTITY_ALIASES = {
    # 公司名变体
    "openai": "OpenAI",
    "deepseek": "DeepSeek",
    "anthropic": "Anthropic",
    "google": "Google",
    "microsoft": "Microsoft",
    "meta": "Meta",
    "字节": "字节跳动",
    "字节跳动": "字节跳动",
    "阿里": "阿里巴巴",
    "阿里巴巴": "阿里巴巴",
    "腾讯": "腾讯",
    "百度": "百度",
    
    # 模型名变体
    "gpt-4": "GPT-4",
    "gpt4": "GPT-4",
    "gpt-3.5": "GPT-3.5",
    "gpt3.5": "GPT-3.5",
    "gpt-5": "GPT-5",
    "gpt5": "GPT-5",
    "gpt-5.4": "GPT-5.4",
    "claude": "Claude",
    "llama": "Llama",
    "llama 3": "Llama 3",
    "llama3": "Llama 3",
    
    # 技术术语
    "transformer": "Transformer",
    "bert": "BERT",
    "gpt": "GPT",
    "moe": "MoE",
    "mixture of experts": "MoE",
}

# 噪音词过滤
NOISE_WORDS = {
    "作者", "编辑", "报道", "一凡", "元宇", "定慧",  # 作者名
    "新智元", "量子位", "qbitai", "公众号",  # 媒体名
    "system-ui", "microsoft yahei", "helvetica",  # CSS
    "点击", "关注", "阅读", "转发", "点赞",  # 操作词
    "原文", "链接", "查看", "更多",  # 导航词
}

# 版本号补全规则
VERSION_PATTERNS = [
    (r'\b(GPT|Claude|Llama|GPT-4|GPT-5)\s*[-]?\s*(\d+(?:\.\d+)?)\b', r'\1-\2'),
    (r'\b(Stable Diffusion)\s*(\d+)\b', r'\1 \2'),
]


def normalize_entity(entity: str) -> str:
    """实体标准化"""
    entity_lower = entity.lower().strip()
    
    # 别名映射
    if entity_lower in ENTITY_ALIASES:
        return ENTITY_ALIASES[entity_lower]
    
    return entity.strip()


def is_noise(entity: str) -> bool:
    """判断是否是噪音"""
    entity_lower = entity.lower()
    
    # 检查噪音词
    for noise in NOISE_WORDS:
        if noise in entity_lower:
            return True
    
    # 检查纯标点或太短
    if len(entity.strip()) <= 1:
        return True
    
    # 检查CSS类名模式
    if re.match(r'^[a-z]+-[a-z]+$', entity_lower):
        return True
    
    return False


def merge_adjacent(entities: List[str], text: str) -> List[str]:
    """合并相邻的实体片段"""
    if not entities:
        return entities
    
    merged = []
    current = entities[0]
    
    for i in range(1, len(entities)):
        next_entity = entities[i]
        
        # 检查是否相邻（中间只有空格或标点）
        pattern = re.escape(current) + r'\s*[-]?\s*' + re.escape(next_entity)
        if re.search(pattern, text, re.IGNORECASE):
            # 合并
            current = current + "-" + next_entity
        else:
            merged.append(current)
            current = next_entity
    
    merged.append(current)
    return merged


def post_process_entities(entities: List[str], text: str) -> List[str]:
    """
    后处理实体列表
    
    Args:
        entities: GLiNER 提取的原始实体
        text: 原文
    
    Returns:
        清洗后的实体列表
    """
    # 1. 过滤噪音
    filtered = [e for e in entities if not is_noise(e)]
    
    # 2. 标准化
    normalized = [normalize_entity(e) for e in filtered]
    
    # 3. 去重
    seen = set()
    unique = []
    for e in normalized:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)
    
    # 4. 合并相邻
    merged = merge_adjacent(unique, text)
    
    return merged
