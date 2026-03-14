"""测试文件"""
import pytest
from keyword_extractor import KeywordExtractor, ExtractorConfig


class TestKeywordExtractor:
    """测试关键词提取器"""
    
    def test_basic_extraction(self):
        """测试基本提取功能"""
        config = ExtractorConfig(model_name="text2vec")
        extractor = KeywordExtractor(config)
        
        text = "人工智能正在改变我们的生活方式，深度学习技术在各个领域得到广泛应用。"
        result = extractor.extract(text, top_k=3)
        
        assert result.method == "keybert"
        assert len(result.keywords) <= 3
        assert result.elapsed_time > 0
        
        for kw in result.keywords:
            assert kw.keyword
            assert 0 <= kw.score <= 1
    
    def test_empty_text(self):
        """测试空文本"""
        config = ExtractorConfig()
        extractor = KeywordExtractor(config)
        
        result = extractor.extract("")
        assert len(result.keywords) == 0
    
    def test_english_keywords(self):
        """测试英文关键词提取"""
        config = ExtractorConfig()
        extractor = KeywordExtractor(config)
        
        text = "OpenAI 发布了 GPT-4 模型，DeepSeek 也推出了新的 AI 产品。"
        result = extractor.extract(text, top_k=5)
        
        # 应该能提取到英文实体
        keywords = [kw.keyword.lower() for kw in result.keywords]
        assert any('openai' in kw or 'gpt' in kw or 'deepseek' in kw for kw in keywords)


class TestExtractorConfig:
    """测试配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ExtractorConfig()
        assert config.top_k == 5
        assert config.model_name == "shibing624/text2vec-base-chinese"
        assert config.use_mmr is True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = ExtractorConfig(
            model_name="bge-m3",
            top_k=10,
            diversity=0.8
        )
        assert config.top_k == 10
        assert config.diversity == 0.8
