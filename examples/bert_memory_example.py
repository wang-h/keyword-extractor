"""
BERT + 双重记忆机制 示例

这个示例展示如何使用 BERT + GTM/ESM 记忆机制处理长文本关键词提取。
相比 KeyBERT，这个方案：
1. 原生支持长文本（通过滑动窗口）
2. 维护全局主题记忆和实体状态记忆
3. 更精准的实体识别（利用 BERT 的深层语义）
"""

from keyword_extractor import BertMemoryExtractor


def example_basic():
    """基础示例"""
    print("=== BERT + 记忆机制 基础示例 ===\n")
    
    # 初始化提取器
    # 首次运行会自动下载模型（约 400MB）
    extractor = BertMemoryExtractor(
        model_name="roberta-wwm",  # 或 "macbert", "tinybert"
        chunk_size=300,            # 每个 chunk 300 tokens
        chunk_overlap=50,          # 重叠 50 tokens
        alpha=0.2,                 # GTM 更新系数
        dynamic_alpha=True         # 首尾段落权重更高
    )
    
    text = """
    OpenAI 近日发布了 GPT-4 Turbo 模型，支持更长的上下文窗口和更低的调用成本。
    与此同时，国内的 DeepSeek 公司也推出了 DeepSeek-V2 模型，在代码生成能力上表现出色。
    字节跳动旗下的豆包大模型团队也在积极布局，推出了针对中文优化的豆包 Pro 版本。
    阿里巴巴的通义千问 2.0 则在多模态能力上有所突破，支持图文理解。
    百度文心一言 4.0 也在持续迭代，增强了推理能力和工具调用功能。
    """
    
    # 提取关键词
    result = extractor.extract(text, top_k=5, return_metadata=True)
    
    print(f"文本长度: {len(text)} 字符")
    print(f"处理时间: {result.elapsed_time:.3f}s")
    print(f"使用模型: {result.model}")
    print("\n提取的关键词:")
    
    for i, kw in enumerate(result.keywords, 1):
        print(f"\n{i}. {kw.keyword}")
        print(f"   得分: {kw.score:.4f}")
        if kw.metadata:
            print(f"   出现次数: {kw.metadata['freq']}")
            print(f"   首次位置: {kw.metadata['first_pos']}")
            print(f"   语义得分: {kw.metadata['component_scores']['semantic']:.4f}")


def example_long_article():
    """长文章示例"""
    print("\n=== 长文章处理示例 ===\n")
    
    extractor = BertMemoryExtractor(
        model_name="roberta-wwm",
        chunk_size=400,
        chunk_overlap=80
    )
    
    # 模拟一篇长文章
    long_text = """
    人工智能技术的快速发展正在重塑各个行业。在这场变革中，英伟达（NVIDIA）凭借其强大的 GPU 
    计算能力成为最大的受益者之一。其 H100 和即将推出的 H200 芯片在训练大语言模型方面表现出色。
    
    与此同时，AMD 也在积极布局 AI 芯片市场，推出了 MI300X 系列加速器，试图挑战英伟达的霸主地位。
    英特尔则推出了 Gaudi 2 和 Gaudi 3 芯片，主打性价比优势。
    
    在模型层面，OpenAI 的 GPT-4 仍然保持着领先地位，但开源模型正在快速追赶。
    Meta 发布的 Llama 3 系列在性能上已经接近 GPT-3.5 水平，而 Mistral AI 的 Mixtral 8x22B 
    则展示了 MoE 架构的潜力。
    
    国内厂商也不甘示弱。DeepSeek 的 DeepSeek-V2 在代码生成方面表现优异，
    月之暗面（Moonshot AI）的 Kimi 则以超长上下文窗口著称，支持高达 200 万字的输入。
    字节跳动的豆包、阿里的通义千问、百度的文心一言都在快速迭代。
    
    在应用层面，ChatGPT 的用户数量已经突破 1.8 亿，成为史上增长最快的消费级应用。
    微软将 Copilot 集成到了 Windows 系统和 Office 套件中，谷歌则在 Bard 基础上推出了 Gemini。
    
    展望未来，多模态模型将成为下一个竞争焦点。GPT-4V、Gemini Pro Vision、
    以及国内的紫东太初、悟道等大模型都在探索图文视频的理解与生成能力。
    具身智能和 AI Agent 也是热门方向，Figure AI、1X Technologies 等公司正在将大模型与机器人结合。
    """ * 3  # 重复 3 次模拟长文
    
    result = extractor.extract(long_text, top_k=10)
    
    print(f"文本长度: {len(long_text)} 字符")
    print(f"处理时间: {result.elapsed_time:.3f}s")
    print("\nTop-10 关键词:")
    
    for i, kw in enumerate(result.keywords, 1):
        print(f"{i:2d}. {kw.keyword:20s} ({kw.score:.4f})")


def example_compare_models():
    """对比不同 BERT 模型"""
    print("\n=== 模型对比示例 ===\n")
    
    text = """
    腾讯发布的混元大模型在中文理解任务上表现出色。
    华为盘古大模型则专注于行业应用，在气象预测、药物研发等领域有所突破。
    京东的言犀大模型主要针对电商场景优化，在商品推荐和客服对话方面效果明显。
    """
    
    models = ["roberta-wwm", "macbert", "tinybert"]
    
    for model_name in models:
        print(f"\n模型: {model_name}")
        try:
            extractor = BertMemoryExtractor(model_name=model_name)
            result = extractor.extract(text, top_k=5)
            print(f"耗时: {result.elapsed_time:.3f}s")
            keywords = ", ".join([kw.keyword for kw in result.keywords])
            print(f"关键词: {keywords}")
        except Exception as e:
            print(f"错误: {e}")


def example_vs_keybert():
    """与 KeyBERT 对比"""
    print("\n=== BERT-Memory vs KeyBERT 对比 ===\n")
    
    from keyword_extractor import KeywordExtractor
    
    text = """
    科技巨头纷纷布局人工智能领域。微软投资了 OpenAI，谷歌推出了 Gemini，
    亚马逊发布了 Amazon Bedrock。国内的阿里巴巴推出通义千问，
    腾讯发布混元大模型，百度持续迭代文心一言。
    创业公司中，月之暗面、智谱 AI、Minimax 等也在快速崛起。
    """
    
    print("测试文本:")
    print(text[:100] + "...\n")
    
    # KeyBERT
    print("KeyBERT 结果:")
    kb_extractor = KeywordExtractor()
    kb_result = kb_extractor.extract(text, top_k=5)
    for kw in kb_result.keywords:
        print(f"  - {kw.keyword} ({kw.score:.4f})")
    
    print(f"  耗时: {kb_result.elapsed_time:.3f}s\n")
    
    # BERT-Memory
    print("BERT-Memory 结果:")
    bert_extractor = BertMemoryExtractor(model_name="tinybert")  # 用轻量版提速
    bert_result = bert_extractor.extract(text, top_k=5)
    for kw in bert_result.keywords:
        print(f"  - {kw.keyword} ({kw.score:.4f})")
    
    print(f"  耗时: {bert_result.elapsed_time:.3f}s")


if __name__ == "__main__":
    example_basic()
    example_long_article()
    # example_compare_models()
    # example_vs_keybert()
