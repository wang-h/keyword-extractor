"""
训练数据生成器 - 从标注数据创建 GLiNER 微调训练集

输入：
- data/articles_full.csv (50篇文章)
- data/article_tags.csv (文章-标签关系)
- data/tags.csv (2618个标签定义)

输出：
- data/gliner_train.jsonl (GLiNER 格式训练数据)
- data/gliner_test.jsonl (测试集)

GLiNER 训练格式：
{
    "text": "文章文本",
    "entities": [
        {"start": 10, "end": 20, "label": "科技公司", "text": "OpenAI"},
        ...
    ]
}
"""
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def load_annotated_data():
    """加载所有标注数据"""
    data_dir = Path(__file__).parent.parent / "data"
    
    # 加载文章
    articles = {}
    with open(data_dir / "articles_full.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            articles[row["id"]] = {
                "title": row["title"],
                "content": row["content"][:4000] if len(row["content"]) > 4000 else row["content"],
            }
    
    # 加载标签映射
    tag_types = {}  # tag_name -> type
    with open(data_dir / "tags.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_types[row["name"]] = row.get("type", "unknown")
    
    # 加载文章-标签关系
    article_tags = defaultdict(list)
    with open(data_dir / "article_tags.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            article_id = row["article_id"]
            tag_name = row["tag_name"]
            tag_type = tag_types.get(tag_name, "unknown")
            article_tags[article_id].append({
                "name": tag_name,
                "type": tag_type
            })
    
    # 合并
    for aid, tags in article_tags.items():
        if aid in articles:
            articles[aid]["tags"] = tags
    
    return {k: v for k, v in articles.items() if "tags" in v}


def map_tag_type_to_label(tag_type: str) -> str:
    """映射标签类型到 GLiNER 标签"""
    type_map = {
        "company": "科技公司全称",
        "product": "软件产品全称",
        "ai_model": "AI模型及版本号",
        "technology": "核心技术术语",
        "hardware": "硬件设备名称",
        "person": "知名人名",
    }
    return type_map.get(tag_type, "技术实体")


def find_entity_positions(text: str, entities: List[Dict]) -> List[Dict]:
    """
    在文本中查找实体位置
    
    返回 GLiNER 格式的实体标注
    """
    text_lower = text.lower()
    found = []
    
    for ent in entities:
        entity_name = ent["name"]
        entity_lower = entity_name.lower()
        
        # 尝试多种匹配方式
        # 1. 精确匹配
        idx = text_lower.find(entity_lower)
        
        # 2. 归一化匹配（去空格、连字符）
        if idx == -1:
            text_norm = text_lower.replace(" ", "").replace("-", "").replace("_", "")
            ent_norm = entity_lower.replace(" ", "").replace("-", "").replace("_", "")
            idx_norm = text_norm.find(ent_norm)
            if idx_norm != -1:
                # 映射回原位置
                idx = idx_norm  # 简化处理
        
        if idx != -1:
            found.append({
                "start": idx,
                "end": idx + len(entity_name),
                "label": map_tag_type_to_label(ent["type"]),
                "text": entity_name
            })
    
    return found


def create_gliner_training_data():
    """创建 GLiNER 格式的训练数据"""
    print("=" * 70)
    print("📚 生成 GLiNER 训练数据")
    print("=" * 70)
    
    articles = load_annotated_data()
    print(f"✅ 加载 {len(articles)} 篇标注文章")
    
    training_examples = []
    
    for aid, article in articles.items():
        # 构建完整文本
        text = f"{article['title']}\n\n{article['content']}"
        
        # 查找实体位置
        entities = find_entity_positions(text, article["tags"])
        
        if entities:  # 只保留有实体标注的
            example = {
                "text": text,
                "entities": entities
            }
            training_examples.append(example)
            print(f"\n📄 {article['title'][:40]}...")
            print(f"   找到 {len(entities)}/{len(article['tags'])} 个实体")
            for e in entities[:3]:
                print(f"   - {e['text']} ({e['label']})")
    
    # 划分训练集/测试集 (80/20)
    split_idx = int(len(training_examples) * 0.8)
    train_data = training_examples[:split_idx]
    test_data = training_examples[split_idx:]
    
    # 保存
    data_dir = Path(__file__).parent.parent / "data"
    
    with open(data_dir / "gliner_train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    with open(data_dir / "gliner_test.jsonl", "w", encoding="utf-8") as f:
        for ex in test_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 70)
    print("📊 训练数据统计")
    print("=" * 70)
    print(f"   训练集: {len(train_data)} 篇")
    print(f"   测试集: {len(test_data)} 篇")
    
    # 统计标签分布
    label_counts = defaultdict(int)
    for ex in training_examples:
        for e in ex["entities"]:
            label_counts[e["label"]] += 1
    
    print(f"\n   标签分布:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"     {label}: {count}")
    
    print(f"\n✅ 已保存到:")
    print(f"   - data/gliner_train.jsonl")
    print(f"   - data/gliner_test.jsonl")


if __name__ == "__main__":
    create_gliner_training_data()
