# Keyword Extractor - 中文关键词提取工具

基于 KeyBERT 的中文关键词提取工具，针对中文文本优化，支持多种 embedding 模型。

## 特性

- 🎯 **中文优化**: 针对中文分词和词性标注优化
- 🤖 **多模型支持**: 支持 text2vec、bge-m3、jina-embeddings 等中文优化模型
- 🧠 **BERT + 记忆机制**: 滑动窗口编码 + 全局主题记忆(GTM) + 实体状态记忆(ESM)，适合长文本精准提取
- ⚡ **轻量级**: 支持 Model2Vec 轻量级模型，CPU 也能快速运行
- 🔧 **灵活配置**: 支持自定义词典、停用词、N-gram 范围等
- 📊 **模型对比**: 内置模型对比功能，方便选择最佳模型
- 🖥️ **CLI 工具**: 命令行交互和批量处理
- 🚀 **API 服务**: 可选 FastAPI 服务（开发中）

## 安装

```bash
# 基础安装（使用 Model2Vec 轻量级模型）
pip install keyword-extractor

# 推荐安装（支持 SentenceTransformers 模型）
pip install "keyword-extractor[hf]"

# 完整安装（包含所有模型和 API 服务）
pip install "keyword-extractor[all]"

# 开发安装
pip install "keyword-extractor[dev]"
```

## 快速开始

### 命令行使用

```bash
# 提取关键词
kwextract extract "人工智能正在改变我们的生活方式"

# 指定模型
kwextract extract "深度学习在图像识别中的应用" -m bge-m3

# 从文件读取
kwextract extract -f article.txt -k 10

# 输出 JSON
kwextract extract "文本内容" --json

# 对比模型
kwextract compare "测试文本" -m text2vec -m bge-m3

# 交互模式
kwextract interactive

# 查看可用模型
kwextract models
```

### Python API

```python
from keyword_extractor import KeywordExtractor, ExtractorConfig

# 默认配置
config = ExtractorConfig()
extractor = KeywordExtractor(config)

# 提取关键词
result = extractor.extract("人工智能正在改变我们的生活方式", top_k=5)

for kw in result.keywords:
    print(f"{kw.keyword}: {kw.score:.4f}")
```

### 高级配置

```python
from keyword_extractor import KeywordExtractor, ExtractorConfig

config = ExtractorConfig(
    model_name="BAAI/bge-m3",  # 模型名称
    top_k=10,                   # 提取数量
    ngram_range=(1, 2),         # N-gram 范围
    diversity=0.7,              # MMR 多样性
    use_mmr=True,               # 使用 MMR
    min_keyword_length=2,       # 最小长度
    max_keyword_length=15,      # 最大长度
    custom_dict="user_dict.txt", # 自定义词典
    stopwords=["停用词1", "停用词2"],  # 自定义停用词
)

extractor = KeywordExtractor(config)
result = extractor.extract(text)
```

## 模型对比

| 模型 | 大小 | 中文优化 | 速度 | 推荐场景 |
|------|------|---------|------|---------|
| text2vec | ~100MB | ✅ | 快 | CPU 环境、轻量级应用 |
| bge-m3 | ~2.2GB | ✅ | 中等 | 高质量需求、多粒度提取 |
| bge-large-zh | ~1.3GB | ✅ | 中等 | 高质量中文 embedding |
| jina-embeddings | ~600MB | ✅ | 快 | 长文本处理 |
| paraphrase-multilingual | ~470MB | ❌ | 快 | 多语言混合文本 |

## 推荐模型

### 轻量级（CPU 友好）
```python
config = ExtractorConfig(model_name="text2vec")
```

### 高质量
```python
config = ExtractorConfig(model_name="bge-m3")
```

## BERT + 记忆机制（长文本方案）

针对微信公众号长文（1000-5000字），提供 BERT + 双重记忆机制方案：

```python
from keyword_extractor import BertMemoryExtractor

extractor = BertMemoryExtractor(
    model_name="roberta-wwm",  # 或 macbert/tinybert
    chunk_size=300,            # 滑动窗口大小
    chunk_overlap=50,          # 窗口重叠
    alpha=0.2,                 # 记忆更新系数
    dynamic_alpha=True         # 动态权重（首尾段落权重更高）
)

# 处理长文本
with open("long_article.txt") as f:
    text = f.read()

result = extractor.extract(text, top_k=10, return_metadata=True)

for kw in result.keywords:
    print(f"{kw.keyword}: {kw.score:.4f}")
    if kw.metadata:
        print(f"  出现次数: {kw.metadata['freq']}")
        print(f"  语义得分: {kw.metadata['component_scores']['semantic']:.4f}")
```

### BERT-Memory vs KeyBERT

| 特性 | KeyBERT | BERT-Memory |
|------|---------|-------------|
| 长文本支持 | 需截断（512 tokens） | ✅ 滑动窗口，完整处理 |
| 全局理解 | ❌ 局部相似度 | ✅ GTM 全局主题记忆 |
| 实体追踪 | ❌ 单次出现 | ✅ ESM 跨窗口实体聚合 |
| 位置感知 | ❌ 无 | ✅ 首尾权重更高 |
| 速度 | ⚡ 快 | 🐢 较慢（BERT 推理） |

### 推荐场景
- **KeyBERT**: 短文本、大规模批量处理、CPU 环境
- **BERT-Memory**: 长文章、需要精准实体识别、可接受秒级延迟

## 工作原理

1. **预处理**: 使用 jieba 进行中文分词和词性标注
2. **候选提取**: 提取名词、专有名词、英文实体作为候选词
3. **Embedding**: 使用预训练模型计算文本和候选词的向量
4. **相似度计算**: 计算候选词与文本的余弦相似度
5. **MMR 多样性**: 使用 Maximal Marginal Relevance 提高多样性

## 与 KeyBERT 的区别

| 特性 | KeyBERT | keyword-extractor |
|------|---------|-------------------|
| 中文分词 | 空格分词 | jieba 分词 + 词性标注 |
| 候选词 | 简单 n-gram | 智能提取（实体 + 短语）|
| 停用词 | 英文为主 | 中文停用词库 |
| 模型选择 | 手动配置 | 预设中文优化模型 |
| 易用性 | 需配置 | 开箱即用 |

## 开发

```bash
# 克隆仓库
git clone https://github.com/wang-h/keyword-extractor.git
cd keyword-extractor

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black src/
ruff check src/
```

## License

MIT License
