"""示例代码"""
from keyword_extractor import KeywordExtractor, ExtractorConfig


def example_basic():
    """基础示例"""
    print("=== 基础示例 ===")
    
    extractor = KeywordExtractor()
    text = "人工智能正在改变我们的生活方式，深度学习技术在各个领域得到广泛应用。"
    
    result = extractor.extract(text, top_k=5)
    
    print(f"文本: {text[:50]}...")
    print(f"方法: {result.method}")
    print(f"耗时: {result.elapsed_time:.3f}s")
    print("关键词:")
    for i, kw in enumerate(result.keywords, 1):
        print(f"  {i}. {kw.keyword} (score: {kw.score:.4f})")


def example_compare_models():
    """对比不同模型"""
    print("\n=== 模型对比示例 ===")
    
    text = "字节跳动发布了豆包视频生成模型，阿里也推出了类似的 AI 产品。"
    
    extractor = KeywordExtractor()
    results = extractor.compare_models(
        text,
        models=["text2vec", "paraphrase-multilingual"],
        top_k=3
    )
    
    for model_name, result in results.items():
        print(f"\n模型: {model_name}")
        if result:
            for kw in result.keywords:
                print(f"  - {kw.keyword} ({kw.score:.4f})")


def example_custom_config():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===")
    
    config = ExtractorConfig(
        model_name="bge-m3",
        top_k=8,
        ngram_range=(1, 3),  # 支持 1-3 个词
        diversity=0.5,  # 降低多样性，更关注相关性
        min_keyword_length=3,
    )
    
    extractor = KeywordExtractor(config)
    
    text = """
    深度学习技术在计算机视觉领域取得了重大突破。
    卷积神经网络（CNN）被广泛应用于图像分类、目标检测等任务。
    近年来，Transformer 架构也被引入视觉领域，产生了 Vision Transformer（ViT）等模型。
    """
    
    result = extractor.extract(text)
    
    print(f"使用模型: {result.model}")
    print("提取的关键词:")
    for kw in result.keywords:
        print(f"  - {kw.keyword}")


def example_batch():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    texts = [
        "OpenAI 发布 GPT-4 Turbo，支持更长的上下文窗口。",
        "DeepSeek 推出新一代大语言模型，性能接近 GPT-4。",
        "阿里巴巴发布通义千问 2.0，中文能力显著提升。",
    ]
    
    extractor = KeywordExtractor()
    results = extractor.extract_batch(texts, top_k=3)
    
    for text, result in zip(texts, results):
        print(f"\n文本: {text[:40]}...")
        keywords = ", ".join([kw.keyword for kw in result.keywords])
        print(f"关键词: {keywords}")


if __name__ == "__main__":
    example_basic()
    example_compare_models()
    example_custom_config()
    example_batch()
