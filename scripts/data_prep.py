#!/usr/bin/env python3
"""
数据准备和增强脚本
"""
import json
import random
import re
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], path: str):
    """保存 JSONL 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def synonym_replace(text: str, entities: List[Dict]) -> List[Dict]:
    """同义词替换增强"""
    augmented = []
    
    # 实体同义词映射
    synonym_map = {
        'OpenAI': ['Open AI', 'openai', 'OpenAi'],
        'GPT-4': ['GPT4', 'gpt-4', 'Gpt-4'],
        'GPT-5': ['GPT5', 'gpt-5', 'Gpt-5'],
        'GPT-5.4': ['GPT5.4', 'gpt-5.4'],
        'ChatGPT': ['Chat GPT', 'chatgpt'],
        'Claude': ['claude', 'Claude AI'],
        'Claude Opus': ['ClaudeOpus', 'claude-opus'],
        'Gemini': ['gemini', 'Gemini AI'],
        'DeepSeek': ['deepseek', 'Deep Seek'],
        'OpenClaw': ['openclaw', 'Openclaw'],
    }
    
    for item in entities:
        ent_text = item['text']
        if ent_text in synonym_map:
            for variant in synonym_map[ent_text]:
                # 替换文本
                new_text = text.replace(ent_text, variant)
                if new_text != text:
                    # 调整实体位置
                    offset = len(variant) - len(ent_text)
                    new_entities = []
                    for e in entities:
                        new_e = e.copy()
                        if e['start'] > item['start']:
                            new_e['start'] += offset
                            new_e['end'] += offset
                        if e['start'] == item['start']:
                            new_e['text'] = variant
                            new_e['end'] = new_e['start'] + len(variant)
                        new_entities.append(new_e)
                    
                    augmented.append({
                        'text': new_text,
                        'entities': new_entities
                    })
    
    return augmented


def add_negative_samples(train_data: List[Dict]) -> List[Dict]:
    """添加负样本（CSS噪音等），统一为 GLiNER JSONL 字段。"""
    negative_texts = [
        "<style>body{font-family: Microsoft YaHei, Arial, sans-serif;}</style>",
        "padding: 20px; margin: 0 auto; background-color: #fff;",
        "作者：编辑 | 来源：量子位公众号",
        "点击图片阅读原文，关注我们的公众号获取更多信息",
    ]

    for neg_text in negative_texts:
        train_data.append({"text": neg_text, "entities": []})

    return train_data


def back_translation_simulation(text: str, entities: List[Dict]) -> Dict:
    """模拟回译增强 (简化为随机词语替换)"""
    # 简化的"回译": 替换一些常见词语
    replacements = {
        '人工智能': 'AI',
        '公司': '企业',
        '发布': '推出',
        '技术': '科技',
        '模型': '大模型',
    }
    
    new_text = text
    for old, new in replacements.items():
        new_text = new_text.replace(old, new)
    
    # 实体位置可能需要调整，简化处理
    return {'text': new_text, 'entities': entities}


def gliner_to_internal(item: Dict) -> Dict:
    """将 GLiNER 格式转换为内部格式"""
    if 'tokenized_text' in item:
        # GLiNER 格式
        entities = []
        for ent in item.get('ner', []):
            if len(ent) >= 3:
                entities.append({
                    'start': ent[0],
                    'end': ent[1],
                    'label': ent[2],
                    'text': item['tokenized_text'][ent[0]:ent[1]]
                })
        return {'text': item['tokenized_text'], 'entities': entities}
    else:
        # 已经是内部格式
        return item


def internal_to_gliner(item: Dict) -> Dict:
    """将内部格式转换为 GLiNER 格式"""
    ner = [[e['start'], e['end'], e['label']] for e in item['entities']]
    return {'tokenized_text': item['text'], 'ner': ner}


def prepare_training_data():
    """主函数：准备训练数据"""
    
    print("数据准备中...")
    
    data_dir = Path("./data")
    train_path = data_dir / "gliner_train_expanded.jsonl"
    if not train_path.exists():
        train_path = data_dir / "gliner_train.jsonl"
    test_path = data_dir / "gliner_test.jsonl"
    
    # 加载原始数据并转换为内部格式
    train_data_raw = load_jsonl(str(train_path))
    test_data_raw = load_jsonl(str(test_path))
    
    train_data = [gliner_to_internal(item) for item in train_data_raw]
    test_data = [gliner_to_internal(item) for item in test_data_raw]
    
    print(f"原始数据: 训练 {len(train_data)} 条, 测试 {len(test_data)} 条")
    
    # 1. 同义词增强
    augmented = []
    for item in train_data:
        if item.get('text') and item.get('entities'):
            augmented.extend(synonym_replace(item['text'], item['entities']))
    
    train_data.extend(augmented)
    print(f"同义词增强后: {len(train_data)} 条")
    
    # 2. 回译模拟
    back_trans = []
    for item in train_data[:20]:  # 只对部分数据进行
        if item.get('text'):
            back_trans.append(back_translation_simulation(item['text'], item.get('entities', [])))
    
    train_data.extend(back_trans)
    print(f"回译增强后: {len(train_data)} 条")
    
    # 3. 添加负样本
    train_data = add_negative_samples(train_data)
    print(f"添加负样本后: {len(train_data)} 条")
    
    # 4. 打乱数据
    random.shuffle(train_data)
    
    # 5. 保存增强后的数据 (转换回 GLiNER 格式)
    augmented_path = data_dir / 'gliner_train_augmented.jsonl'
    train_data_gliner = [internal_to_gliner(item) for item in train_data]
    save_jsonl(train_data_gliner, str(augmented_path))
    print(f"增强数据已保存: {augmented_path}")
    
    # 统计
    total_entities = sum(len(item['entities']) for item in train_data)
    print(f"\n统计:")
    print(f"  训练样本: {len(train_data)}")
    print(f"  实体总数: {total_entities}")
    print(f"  平均每篇实体: {total_entities/len(train_data):.1f}")


if __name__ == "__main__":
    prepare_training_data()
