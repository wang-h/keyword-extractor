"""
MLX-LLM 本地大模型关键词提取器

基于 Apple MLX 框架的本地 LLM 提取方案
优势：
- 原生支持 Apple Silicon GPU/ANE
- 长文本无需分块（32K+ 上下文）
- 通过 Prompt 精准控制实体类型
"""
import json
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .models import KeywordItem, ExtractionResult


@dataclass
class MlxLLMConfig:
    """MLX LLM 配置"""
    model_name: str = "qwen3-4b"  # 或 qwen3-3b, qwen3-1.5b
    max_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    
    # 提示词配置
    system_prompt: str = """你是一个专业的科技情报分析引擎，擅长从文章中提取核心关键词。
要求：
1. 严格只提取：公司名称、产品名称、核心技术名称
2. 按重要程度排序
3. 必须以严格的 JSON 格式输出
4. 每个类别最多 5 个，总数不超过 10 个"""
    
    user_prompt_template: str = """请从以下文章中提取核心关键词。

文章内容：
{text}

请以 JSON 格式输出：
{{
  "company": ["公司1", "公司2"],
  "product": ["产品1", "产品2"],
  "technology": ["技术1", "技术2"]
}}"""


class MlxLLMExtractor:
    """MLX LLM 关键词提取器"""
    
    # 预设模型映射
    PRESET_MODELS = {
        "qwen3-1.5b": "qwen/Qwen3-1.5B-Instruct-4bit",
        "qwen3-3b": "qwen/Qwen3-3B-Instruct-4bit",
        "qwen3-4b": "qwen/Qwen3-4B-Instruct-4bit",
    }
    
    def __init__(self, config: Optional[MlxLLMConfig] = None):
        """
        初始化提取器
        
        Args:
            config: 配置对象
        """
        self.config = config or MlxLLMConfig()
        self._model = None
        self._tokenizer = None
        
        # 检查 mlx-lm 是否安装
        try:
            from mlx_lm import load, generate
            self._mlx_available = True
        except ImportError:
            self._mlx_available = False
            logger.warning("mlx-lm 未安装，请运行: pip install mlx-lm")
    
    def _load_model(self):
        """懒加载模型"""
        if self._model is not None or not self._mlx_available:
            return
        
        from mlx_lm import load
        
        model_name = self.config.model_name
        if model_name in self.PRESET_MODELS:
            model_name = self.PRESET_MODELS[model_name]
        
        logger.info(f"正在加载 MLX 模型: {model_name}")
        start = time.time()
        
        self._model, self._tokenizer = load(model_name)
        
        logger.info(f"✅ 模型加载完成，耗时: {time.time()-start:.2f}s")
    
    def extract(
        self,
        text: str,
        top_k: int = 10
    ) -> ExtractionResult:
        """
        提取关键词
        
        Args:
            text: 输入文本
            top_k: 返回关键词数量（LLM 自动控制，此参数用于截取）
        
        Returns:
            提取结果
        """
        start_time = time.time()
        
        if not text or not text.strip():
            return ExtractionResult(
                text="",
                keywords=[],
                method="mlx-llm",
                elapsed_time=0.0,
                model=self.config.model_name
            )
        
        if not self._mlx_available:
            logger.error("mlx-lm 未安装")
            return ExtractionResult(
                text=text[:200],
                keywords=[],
                method="error",
                elapsed_time=time.time() - start_time,
                model=None
            )
        
        # 加载模型
        self._load_model()
        
        # 构造提示词
        user_prompt = self.config.user_prompt_template.format(text=text)
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 使用 chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        logger.debug(f"Prompt length: {len(prompt)} chars")
        
        # 生成
        from mlx_lm import generate
        
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            verbose=False
        )
        
        # 解析 JSON
        keywords = self._parse_response(response)
        
        elapsed = time.time() - start_time
        
        return ExtractionResult(
            text=text[:200] + "..." if len(text) > 200 else text,
            keywords=keywords[:top_k],
            method="mlx-llm",
            elapsed_time=elapsed,
            model=self.config.model_name
        )
    
    def _parse_response(self, response: str) -> List[KeywordItem]:
        """解析 LLM 返回的 JSON"""
        keywords = []
        
        try:
            # 尝试提取 JSON 块
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                # 处理不同格式
                if isinstance(data, dict):
                    # 格式: {"company": [...], "product": [...]}
                    for category, items in data.items():
                        for item in items:
                            if isinstance(item, str):
                                keywords.append(KeywordItem(
                                    keyword=item,
                                    score=1.0,  # LLM 输出的不计算分数
                                    method="mlx-llm",
                                    metadata={"category": category}
                                ))
                elif isinstance(data, list):
                    # 格式: ["关键词1", "关键词2"]
                    for item in data:
                        if isinstance(item, str):
                            keywords.append(KeywordItem(
                                keyword=item,
                                score=1.0,
                                method="mlx-llm"
                            ))
        except Exception as e:
            logger.warning(f"JSON 解析失败: {e}, 原始响应: {response[:500]}")
            # 回退：按行解析
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.startswith('{') and not line.startswith('}'):
                    # 清理常见的 JSON 符号
                    line = line.strip('",[]')
                    if line and len(line) > 1:
                        keywords.append(KeywordItem(
                            keyword=line,
                            score=0.5,
                            method="mlx-llm-fallback"
                        ))
        
        return keywords
    
    def extract_batch(
        self,
        texts: List[str],
        top_k: int = 10
    ) -> List[ExtractionResult]:
        """批量提取"""
        return [self.extract(text, top_k) for text in texts]
