"""数据模型"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class KeywordItem(BaseModel):
    """单个关键词/实体结果"""
    keyword: str = Field(..., description="关键词文本")
    score: float = Field(..., description="置信度分数 (0-1)")
    method: str = Field(..., description="提取方法标识")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")


class ExtractionResult(BaseModel):
    """关键词提取批结果"""
    text: str = Field(..., description="原始文本（截断）")
    keywords: List[KeywordItem] = Field(default_factory=list)
    method: str = Field(..., description="主要提取方法")
    elapsed_time: float = Field(..., description="耗时（秒）")
    model: Optional[str] = Field(None, description="所用模型名称")
