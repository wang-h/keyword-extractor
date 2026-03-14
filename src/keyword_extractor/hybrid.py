"""
Hybrid 实体提取器 - Gazetteer + GLiNER + 双重记忆

架构：
1. Gazetteer Match: 从标签库精确匹配（高 Precision）
2. GLiNER Recall: 零-shot 实体识别（高 Recall）
3. Dual-Memory Fusion: 跨段落全局融合
4. Post-process: 去噪、归一化、重排序
"""
import re
import time
from typing import List, Dict, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from .gliner_memory import GLiNEREntityExtractor
from .gazetteer import GazetteerMatcher
from .postprocess import post_process_entities
from .models import ExtractionResult, KeywordItem


class HybridEntityExtractor:
    """
    混合实体提取器
    
    结合 Gazetteer 的精确性和 GLiNER 的泛化能力
    """
    
    def __init__(
        self,
        gazetteer_path: str = None,
        gliner_model: str = "urchade/gliner_multi-v2.1",
        labels: List[str] = None,
        chunk_size: int = 1000,
        threshold: float = 0.25,
        gazetteer_weight: float = 0.6,  # Gazetteer 匹配权重
        gliner_weight: float = 0.4,     # GLiNER 匹配权重
    ):
        """
        初始化 Hybrid 提取器
        
        Args:
            gazetteer_path: tags.csv 路径
            gliner_model: GLiNER 模型名
            labels: GLiNER 标签定义
            gazetteer_weight: Gazetteer 结果权重
            gliner_weight: GLiNER 结果权重
        """
        # 默认标签
        if labels is None:
            labels = [
                "科技公司名称",
                "AI软件产品全称",
                "大语言模型及版本号",
                "人名",
                "核心技术术语"
            ]
        
        self.gazetteer = GazetteerMatcher(gazetteer_path)
        self.gliner = GLiNEREntityExtractor(
            model_name=gliner_model,
            labels=labels,
            chunk_size=chunk_size,
            threshold=threshold
        )
        
        self.gazetteer_weight = gazetteer_weight
        self.gliner_weight = gliner_weight
        
        logger.info(f"Hybrid 提取器初始化完成")
        logger.info(f"  Gazetteer 权重: {gazetteer_weight}")
        logger.info(f"  GLiNER 权重: {gliner_weight}")
    
    def extract(
        self,
        text: str,
        title: str = "",
        top_k: int = 10,
        return_metadata: bool = False
    ) -> ExtractionResult:
        """
        提取实体
        
        Args:
            text: 正文内容
            title: 标题（用于 Gazetteer 标题匹配加权）
            top_k: 返回数量
            return_metadata: 是否返回详细元数据
        
        Returns:
            提取结果
        """
        start_time = time.time()
        
        # 1. Gazetteer 匹配（精确匹配层）
        gazetteer_matches = self.gazetteer.match(text, title, top_k=top_k*2)
        gazetteer_entities = {m.entity: m.score * self.gazetteer_weight for m in gazetteer_matches}
        
        # 2. GLiNER 识别（泛化召回层）
        # 使用标题+正文
        full_text = f"{title}\n\n{text}" if title else text
        gliner_result = self.gliner.extract(full_text, top_k=top_k*2)
        gliner_entities = {}
        for kw in gliner_result.keywords:
            entity = kw.keyword.split(" (")[0]  # 去掉标签类型
            gliner_entities[entity] = kw.score * self.gliner_weight
        
        # 3. 融合策略
        # 3.1 合并实体集合
        all_entities = set(gazetteer_entities.keys()) | set(gliner_entities.keys())
        
        # 3.2 加权融合分数
        fused_scores = {}
        for entity in all_entities:
            g_score = gazetteer_entities.get(entity, 0.0)
            gl_score = gliner_entities.get(entity, 0.0)
            
            # 融合公式：加权平均 + 两者都命中时的 bonus
            fused_scores[entity] = max(g_score, gl_score)
            if g_score > 0 and gl_score > 0:
                # 两者都命中，加分
                fused_scores[entity] += 0.1
        
        # 4. 后处理
        # 4.1 过滤噪音
        filtered = {e: s for e, s in fused_scores.items() if not self._is_noise(e)}
        
        # 4.2 归一化
        normalized = self._normalize_entities(filtered)
        
        # 4.3 排序取 top_k
        sorted_entities = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        final_entities = sorted_entities[:top_k]
        
        # 5. 构建结果
        keywords = [
            KeywordItem(
                keyword=entity,
                score=score,
                source="hybrid",
                method="hybrid-gazetteer-gliner"
            )
            for entity, score in final_entities
        ]
        
        elapsed = time.time() - start_time
        
        return ExtractionResult(
            text=text[:200],
            keywords=keywords,
            method="hybrid-gazetteer-gliner",
            elapsed_time=elapsed,
            model=self.gliner.model_name
        )
    
    def _is_noise(self, entity: str) -> bool:
        """判断是否是噪音"""
        # 太短
        if len(entity) <= 1:
            return True
        
        # 纯数字
        if entity.replace('.', '').replace('-', '').isdigit():
            return True
        
        # 常见噪音词
        noise_words = {'作者', '编辑', '报道', '新智元', '量子位', '公众号', '点击'}
        if any(w in entity for w in noise_words):
            return True
        
        return False
    
    def _normalize_entities(self, entities: Dict[str, float]) -> Dict[str, float]:
        """实体归一化（合并相似实体）"""
        # 简单的归一化规则
        normalized = {}
        
        for entity, score in entities.items():
            # 统一大小写（保留首字母大写）
            if entity.isupper() and len(entity) <= 4:
                # 缩写保持大写
                norm = entity
            else:
                # 其他首字母大写
                norm = entity.title() if entity.islower() else entity
            
            # 合并相同归一化形式的实体，取最高分
            if norm in normalized:
                normalized[norm] = max(normalized[norm], score)
            else:
                normalized[norm] = score
        
        return normalized
