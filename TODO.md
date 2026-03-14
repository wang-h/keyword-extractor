# Keyword Extractor 项目待办清单

## 当前状态
- ✅ 基础项目结构
- ✅ KeyBERT 轻量方案
- ✅ BERT + GTM/ESM 双重记忆机制
- ✅ GitHub 推送 (https://github.com/wang-h/keyword-extractor)

## 已完成 ✅

### P0 - 核心验证
- ✅ 1. 本地测试 MacBERT 提取效果
  - MacBERT 模型加载成功
  - 长文本分块处理正常 (5 chunks)
  - GTM/ESM 记忆机制工作正常

### P1 - 功能增强
- ✅ 2. 添加 LLM 方案 (mlx-lm + Qwen3-4B)
  - 实现 MlxLLMExtractor 类
  - 支持 qwen3-1.5b/3b/4b 量化版本
  - JSON 解析和错误回退

### P2 - 工程落地
- ✅ 3. 集成到 werss 项目
  - 修改 core/tag_extractor.py
  - 添加 extract_with_bert_memory 方法
  - 支持 method=bert-memory 和 method=auto
  - 自动路由：长文本(>1000字)用 BERT-Memory，短文本用 KeyBERT

### P3 - 发布 (部分完成)
- ✅ GitHub 推送
- ⬜ PyPI 发布
- ⬜ 完善文档

## 仓库地址
- **keyword-extractor**: https://github.com/wang-h/keyword-extractor
  - BERT-Memory + MLX-LLM 完整实现
  - 三种方案：KeyBERT / BERT-Memory / MLX-LLM
  
- **werss**: 本地已集成（待 push）
  - 自动路由策略
  - 配置: `article_tag.extract_method=auto`

## 使用方式

### keyword-extractor 独立使用
```python
from keyword_extractor import BertMemoryExtractor, MlxLLMExtractor

# BERT-Memory (长文本，高精度)
extractor = BertMemoryExtractor(model_name='macbert')
result = extractor.extract(long_text, top_k=10)

# MLX-LLM (Apple Silicon 本地大模型)
extractor = MlxLLMExtractor(model_name='qwen3-4b')
result = extractor.extract(text, top_k=10)
```

### werss 集成使用
```python
# 配置 config.yaml
article_tag:
  extract_method: auto  # 自动路由
  # 或
  extract_method: bert-memory  # 强制使用 BERT
  bert:
    model: macbert  # macbert/roberta-wwm/tinybert
```

## 下一步（可选）
- [ ] PyPI 发布 pip install keyword-extractor
- [ ] werss 推送认证修复
- [ ] 补充性能测试报告
- [ ] 接入 werss 生产环境测试
