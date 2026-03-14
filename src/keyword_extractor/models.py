"""数据模型定义"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class KeywordItem(BaseModel):
    """单个关键词结果"""
    keyword: str = Field(..., description="关键词")
    score: float = Field(..., description="相关性分数 (0-1)")
    method: str = Field(..., description="提取方法")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")


class ExtractionResult(BaseModel):
    """关键词提取结果"""
    text: str = Field(..., description="原始文本（截断）")
    keywords: List[KeywordItem] = Field(default_factory=list, description="提取的关键词列表")
    method: str = Field(..., description="使用的主要方法")
    elapsed_time: float = Field(..., description="耗时（秒）")
    model: Optional[str] = Field(None, description="使用的模型名称")


class ExtractorConfig(BaseModel):
    """提取器配置"""
    # 模型配置
    model_name: str = Field(
        default="shibing624/text2vec-base-chinese",
        description="Embedding 模型名称"
    )
    device: str = Field(default="auto", description="运行设备 (cpu/cuda/auto)")
    
    # 提取参数
    top_k: int = Field(default=5, ge=1, le=20, description="提取关键词数量")
    ngram_range: tuple = Field(default=(1, 2), description="N-gram 范围")
    diversity: float = Field(default=0.7, ge=0.0, le=1.0, description="MMR 多样性参数")
    use_mmr: bool = Field(default=True, description="是否使用 MMR")
    
    # 中文分词
    custom_dict: Optional[str] = Field(None, description="自定义词典路径")
    stopwords: Optional[List[str]] = Field(None, description="停用词列表")
    
    # LLM 配置
    llm_backend: Literal["openai", "ollama", "none"] = Field(
        default="none",
        description="LLM 后端类型"
    )
    llm_model: Optional[str] = Field(None, description="LLM 模型名称")
    llm_api_key: Optional[str] = Field(None, description="API Key")
    llm_base_url: Optional[str] = Field(None, description="API Base URL")
    
    # 混合策略
    use_hybrid: bool = Field(default=True, description="是否使用混合策略（KeyBERT + LLM）")
    fallback_to_llm: bool = Field(default=True, description="KeyBERT 效果差时回退到 LLM")
    
    # 预处理
    min_keyword_length: int = Field(default=2, ge=1, description="最小关键词长度")
    max_keyword_length: int = Field(default=15, ge=1, description="最大关键词长度")
    filter_numbers: bool = Field(default=True, description="是否过滤纯数字")
    
    class Config:
        extra = "allow"


class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    description: str
    size: str
    chinese_optimized: bool
    multilingual: bool
    recommended: bool = False


# 预定义模型列表
PRESET_MODELS: Dict[str, ModelInfo] = {
    "text2vec": ModelInfo(
        name="shibing624/text2vec-base-chinese",
        description="轻量级中文句向量模型，适合 CPU",
        size="~100MB",
        chinese_optimized=True,
        multilingual=False,
        recommended=True,
    ),
    "bge-m3": ModelInfo(
        name="BAAI/bge-m3",
        description="BGE 多语言模型，支持多粒度（单词、短语、句子、文档）",
        size="~2.2GB",
        chinese_optimized=True,
        multilingual=True,
        recommended=True,
    ),
    "bge-large-zh": ModelInfo(
        name="BAAI/bge-large-zh-v1.5",
        description="BGE 中文大模型，高质量 embedding",
        size="~1.3GB",
        chinese_optimized=True,
        multilingual=False,
    ),
    "jina-embeddings": ModelInfo(
        name="jinaai/jina-embeddings-v3",
        description="Jina 多语言嵌入模型，支持长文本",
        size="~600MB",
        chinese_optimized=True,
        multilingual=True,
    ),
    "paraphrase-multilingual": ModelInfo(
        name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="通用多语言模型",
        size="~470MB",
        chinese_optimized=False,
        multilingual=True,
    ),
}
