# Keyword Extractor 项目待办清单

## 当前状态
- ✅ 基础项目结构
- ✅ KeyBERT 轻量方案
- ✅ BERT + GTM/ESM 双重记忆机制
- ✅ GitHub 推送 (https://github.com/wang-h/keyword-extractor)

## 待办任务

### P0 - 核心验证
- [ ] 1. 本地测试 MacBERT 提取效果
  - 安装依赖并运行示例
  - 验证长文本处理 (>3000字)
  - 对比 KeyBERT vs BERT-Memory 效果

### P1 - 功能增强
- [ ] 2. 添加 LLM 方案 (mlx-lm + Qwen3-4B)
  - 实现 MlxLLMExtractor 类
  - 支持本地部署 Qwen3-4B/3B/1.5B
  - 对比 LLM vs BERT 效果

### P2 - 工程化
- [ ] 3. 集成到 werss 项目
  - 替换 core/tag_extractor.py
  - 配置切换逻辑 (短文本用 KeyBERT, 长文用 BERT/LLM)
  - 测试微信公众号文章提取

### P3 - 发布
- [ ] 4. PyPI 发布
  - 完善 pyproject.toml
  - 打 tag (v0.1.0)
  - 上传至 PyPI

- [ ] 5. 完善文档
  - 技术文档 (GTM/ESM 原理详解)
  - API 文档
  - 性能对比报告

## 当前执行
任务 1: 本地测试 MacBERT
