"""
Gazetteer 实体匹配器 - 基于标签库的精确匹配

用途：
1. 从已标注的 2618 个标签中匹配实体
2. 支持精确匹配、归一化匹配、模糊匹配
3. 作为 GLiNER 的补充召回层
"""
import re
import csv
from typing import List, Dict, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
import difflib


@dataclass
class MatchResult:
    entity: str
    match_type: str  # 'exact', 'normalized', 'fuzzy', 'title'
    score: float
    position: int = -1


class GazetteerMatcher:
    """基于标签库的实体匹配器"""
    
    def __init__(self, tags_file: str = None):
        """
        初始化 Gazetteer
        
        Args:
            tags_file: tags.csv 文件路径
        """
        self.tags: Set[str] = set()
        self.tag_aliases: Dict[str, str] = {}  # 别名 -> 标准名
        
        if tags_file and Path(tags_file).exists():
            self._load_tags(tags_file)
        
        # 构建归一化映射
        self.normalized_map: Dict[str, str] = {}
        for tag in self.tags:
            norm = self._normalize(tag)
            self.normalized_map[norm] = tag
    
    def _load_tags(self, filepath: str):
        """从 CSV 加载标签"""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = row.get('name', '').strip()
                if tag:
                    self.tags.add(tag)
    
    def _normalize(self, text: str) -> str:
        """文本归一化"""
        text = text.lower()
        # 移除空格、连字符、下划线、括号
        text = re.sub(r'[\s\-_()（）]', '', text)
        # 统一版本号格式
        text = re.sub(r'(gpt|claude|llama|gemini)\s*(\d)', r'\1\2', text)
        return text
    
    def _extract_versioned_entities(self, text: str) -> List[Tuple[str, int]]:
        """提取带版本号的实体，如 GPT-5.4"""
        # 匹配模式：Name + 数字/版本号
        pattern = r'\b(OpenAI|DeepSeek|Anthropic|Google|Meta|Microsoft|字节跳动|阿里巴巴|腾讯|百度|GPT|Claude|Llama|Gemini|Kimi|豆包|Codex)(?:\s*[-]?\s*(\d+(?:\.\d+)?))?\b'
        matches = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entity = match.group(1)
            version = match.group(2)
            if version:
                entity = f"{entity}-{version}"
            matches.append((entity, match.start()))
        return matches
    
    def match(self, text: str, title: str = "", top_k: int = 10) -> List[MatchResult]:
        """
        在文本中匹配标签库实体
        
        Args:
            text: 正文内容
            title: 标题（标题匹配权重更高）
            top_k: 返回数量
        
        Returns:
            匹配结果列表
        """
        results = []
        text_lower = text.lower()
        title_lower = title.lower()
        
        # 1. 精确匹配（标题中）
        for tag in self.tags:
            tag_lower = tag.lower()
            if tag_lower in title_lower:
                pos = title_lower.find(tag_lower)
                results.append(MatchResult(tag, 'title', 1.0, pos))
        
        # 2. 精确匹配（正文中）
        for tag in self.tags:
            tag_lower = tag.lower()
            if tag_lower in text_lower:
                # 检查是否已在标题中匹配
                if not any(r.entity == tag for r in results):
                    pos = text_lower.find(tag_lower)
                    results.append(MatchResult(tag, 'exact', 0.9, pos))
        
        # 3. 归一化匹配
        text_norm = self._normalize(text)
        for norm, original in self.normalized_map.items():
            if norm in text_norm and norm not in [self._normalize(r.entity) for r in results]:
                results.append(MatchResult(original, 'normalized', 0.8))
        
        # 4. 版本号实体提取
        versioned = self._extract_versioned_entities(text)
        for entity, pos in versioned:
            norm_entity = self._normalize(entity)
            # 检查是否是已知标签
            if norm_entity in self.normalized_map:
                original = self.normalized_map[norm_entity]
                if not any(r.entity == original for r in results):
                    results.append(MatchResult(original, 'exact', 0.95, pos))
            else:
                # 新实体（可能是新版本）
                results.append(MatchResult(entity, 'versioned', 0.7, pos))
        
        # 去重并排序
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x.score, reverse=True):
            key = self._normalize(r.entity)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        
        return unique[:top_k]
    
    def get_candidates(self, text: str, title: str = "") -> List[str]:
        """获取匹配到的实体名称列表"""
        matches = self.match(text, title)
        return [m.entity for m in matches]
