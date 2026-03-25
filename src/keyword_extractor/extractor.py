"""核心关键词提取器"""
import re
import time
from typing import List, Optional, Callable, Any
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import CountVectorizer
from loguru import logger

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    logger.warning("KeyBERT 未安装，部分功能不可用")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from model2vec import Model2Vec
    MODEL2VEC_AVAILABLE = True
except ImportError:
    MODEL2VEC_AVAILABLE = False

from .models import KeywordItem, ExtractionResult, ExtractorConfig, PRESET_MODELS
from .stopwords import DEFAULT_STOPWORDS


class KeywordExtractor:
    """中文关键词提取器"""
    
    def __init__(self, config: Optional[ExtractorConfig] = None):
        """
        初始化提取器
        
        Args:
            config: 配置对象，如果为 None 则使用默认配置
        """
        self.config = config or ExtractorConfig()
        self._model = None
        self._keybert = None
        
        # 加载自定义词典
        if self.config.custom_dict:
            jieba.load_userdict(self.config.custom_dict)
        
        # 初始化停用词
        self._stopwords = set(self.config.stopwords or DEFAULT_STOPWORDS)
        
        logger.info(f"KeywordExtractor 初始化完成，模型: {self.config.model_name}")
    
    def _load_model(self):
        """懒加载模型"""
        if self._model is not None:
            return
        
        model_name = self.config.model_name
        
        # 检查是否是预设模型
        if model_name in PRESET_MODELS:
            model_name = PRESET_MODELS[model_name].name
        
        logger.info(f"正在加载模型: {model_name}")
        start_time = time.time()
        
        # 优先使用 sentence-transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._model = SentenceTransformer(model_name, device=self.config.device)
            logger.info(f"✅ 已加载 SentenceTransformer 模型 (耗时: {time.time() - start_time:.2f}s)")
        # 备选：Model2Vec
        elif MODEL2VEC_AVAILABLE:
            self._model = Model2Vec(model_name)
            logger.info(f"✅ 已加载 Model2Vec 模型 (耗时: {time.time() - start_time:.2f}s)")
        else:
            raise RuntimeError(
                "没有可用的 embedding 后端，请安装: "
                "pip install sentence-transformers 或 pip install model2vec"
            )
        
        # 初始化 KeyBERT
        self._keybert = KeyBERT(model=self._model)
    
    def _chinese_tokenizer(self, text: str) -> List[str]:
        """
        中文分词器
        
        同时处理中文和英文，保留英文单词完整性
        """
        tokens = []
        for token in jieba.cut(text):
            token = token.strip()
            if not token:
                continue
            
            # 纯英文单词（可能包含数字、连字符）
            if re.match(r'^[A-Za-z0-9\-_.]+$', token):
                tokens.append(token.lower())
            # 包含中文
            elif re.search(r'[\u4e00-\u9fa5]', token):
                tokens.append(token)
            # 其他（保留有意义的字符）
            elif not re.match(r'^[^\w\u4e00-\u9fa5]+$', token):
                tokens.append(token)
        
        return tokens
    
    def _extract_candidates(self, text: str) -> List[str]:
        """
        提取候选关键词
        
        使用 jieba 词性标注提取名词、专有名词等
        """
        candidates = []
        
        # 提取专有名词和实体
        words = list(pseg.cut(text))
        for word, flag in words:
            # nr: 人名, nt: 机构名, nz: 其他专名, n: 名词
            if flag in ['nr', 'nt', 'nz', 'n', 'ns']:
                if self._is_valid_keyword(word):
                    candidates.append(word)
        
        # 提取英文专有名词（首字母大写）
        english_pattern = r'([A-Z][a-zA-Z0-9]*(?:\-[A-Z]?[a-zA-Z0-9]*)*)'
        english_matches = re.findall(english_pattern, text)
        for match in english_matches:
            if self._is_valid_keyword(match):
                candidates.append(match)
        
        # 提取短语（2-3 个词的组合）
        for i in range(len(words) - 1):
            # 2-gram
            w1, f1 = words[i]
            w2, f2 = words[i + 1]
            if f1 in ['n', 'nz', 'a'] and f2 in ['n', 'nz']:
                phrase = w1 + w2
                if self._is_valid_keyword(phrase):
                    candidates.append(phrase)
            
            # 3-gram
            if i < len(words) - 2:
                w3, f3 = words[i + 2]
                if f1 in ['n', 'nz'] and f2 in ['n', 'nz'] and f3 in ['n', 'nz']:
                    phrase = w1 + w2 + w3
                    if self._is_valid_keyword(phrase):
                        candidates.append(phrase)
        
        # 去重
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique_candidates.append(c)
        
        return unique_candidates
    
    def _is_valid_keyword(self, keyword: str) -> bool:
        """检查关键词是否有效"""
        keyword = keyword.strip()
        
        # 长度检查
        if len(keyword) < self.config.min_keyword_length:
            return False
        if len(keyword) > self.config.max_keyword_length:
            return False
        
        # 停用词检查
        if keyword in self._stopwords:
            return False
        
        # 纯数字检查
        if self.config.filter_numbers and keyword.isdigit():
            return False
        
        # 标点符号检查
        if re.search(r'[，。、；：！？,\.;:!?（）【】《》"\'「」『』]', keyword):
            return False
        
        return True
    
    def _filter_keywords(self, keywords: List[tuple]) -> List[KeywordItem]:
        """过滤并格式化关键词"""
        result = []
        seen = set()
        
        for item in keywords:
            if isinstance(item, tuple):
                kw, score = item
            else:
                kw, score = item, 0.0
            
            kw = str(kw).strip().replace(' ', '')
            
            # 去重
            kw_lower = kw.lower()
            if kw_lower in seen:
                continue
            seen.add(kw_lower)
            
            # 验证
            if not self._is_valid_keyword(kw):
                continue
            
            result.append(KeywordItem(
                keyword=kw,
                score=float(score),
                method="keybert"
            ))
        
        return result
    
    def extract(
        self,
        text: str,
        top_k: Optional[int] = None
    ) -> ExtractionResult:
        """
        提取关键词
        
        Args:
            text: 输入文本
            top_k: 提取数量，如果为 None 则使用配置中的值
        
        Returns:
            提取结果
        """
        start_time = time.time()
        top_k = top_k or self.config.top_k
        
        if not text or not text.strip():
            return ExtractionResult(
                text="",
                keywords=[],
                method="none",
                elapsed_time=0.0
            )
        
        # 清理文本
        text = text.strip()
        
        try:
            # 加载模型
            self._load_model()
            
            # 创建自定义 Vectorizer
            vectorizer = CountVectorizer(
                tokenizer=self._chinese_tokenizer,
                ngram_range=self.config.ngram_range,
                stop_words=list(self._stopwords),
                max_features=1000
            )
            
            # 提取候选词
            candidates = self._extract_candidates(text)
            
            # 使用 KeyBERT 提取
            if candidates:
                # 如果有候选词，使用候选词模式
                keywords = self._keybert.extract_keywords(
                    text,
                    candidates=candidates,
                    vectorizer=vectorizer,
                    top_n=min(top_k * 3, len(candidates)),
                    use_mmr=self.config.use_mmr,
                    diversity=self.config.diversity
                )
            else:
                # 无候选词时使用标准模式
                keywords = self._keybert.extract_keywords(
                    text,
                    vectorizer=vectorizer,
                    top_n=top_k * 3,
                    use_mmr=self.config.use_mmr,
                    diversity=self.config.diversity
                )
            
            # 过滤并格式化
            filtered = self._filter_keywords(keywords)
            
            return ExtractionResult(
                text=text[:200] + "..." if len(text) > 200 else text,
                keywords=filtered[:top_k],
                method="keybert",
                elapsed_time=time.time() - start_time,
                model=self.config.model_name
            )
            
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return ExtractionResult(
                text=text[:200] + "..." if len(text) > 200 else text,
                keywords=[],
                method="error",
                elapsed_time=time.time() - start_time,
                model=None
            )
    
    def extract_batch(
        self,
        texts: List[str],
        top_k: Optional[int] = None
    ) -> List[ExtractionResult]:
        """批量提取关键词"""
        return [self.extract(text, top_k) for text in texts]
    
    def compare_models(
        self,
        text: str,
        models: Optional[List[str]] = None,
        top_k: int = 5
    ) -> dict:
        """
        对比不同模型的提取效果
        
        Args:
            text: 测试文本
            models: 模型列表，如果为 None 则使用预设模型
            top_k: 提取数量
        
        Returns:
            各模型的提取结果
        """
        models = models or list(PRESET_MODELS.keys())
        results = {}
        
        original_model = self.config.model_name
        
        for model_key in models:
            if model_key not in PRESET_MODELS:
                continue
            
            logger.info(f"测试模型: {model_key}")
            self.config.model_name = model_key
            self._model = None  # 强制重新加载
            self._keybert = None
            
            try:
                result = self.extract(text, top_k)
                results[model_key] = result
            except Exception as e:
                logger.error(f"模型 {model_key} 测试失败: {e}")
                results[model_key] = None
        
        # 恢复原模型
        self.config.model_name = original_model
        self._model = None
        self._keybert = None
        
        return results
