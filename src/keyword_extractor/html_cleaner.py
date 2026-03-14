"""
HTML 清洗工具 - 专门针对微信公众号文章优化

问题：原始 HTML 包含大量 CSS 样式代码，干扰实体提取
解决：多层过滤，保留纯文本内容
"""
import re
from typing import Optional

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


# CSS 相关关键词黑名单
CSS_KEYWORDS = {
    # 字体名称
    'helvetica', 'arial', 'microsoft yahei', 'microsoft yahei ui',
    'hiragino sans gb', 'pingfang sc', 'noto sans', 'sans-serif',
    'serif', 'monospace', 'courier new', 'times new roman',
    'wenquanyi micro hei', 'stheiti', 'stheiti light',
    
    # CSS 属性
    'padding', 'margin', 'font-size', 'line-height', 'color:',
    'background', 'border', 'display:', 'width:', 'height:',
    'text-align', 'vertical-align', 'overflow', 'position:',
    'z-index', 'float:', 'clear:', 'opacity', 'visibility',
    
    # HTML 标签/类名常见噪音
    'mp;', 'data-', 'aria-', 'role=', 'tabindex', 'draggable',
    'contenteditable', 'spellcheck', 'translate', 'dir=',
    
    # 微信特有
    'rich_media', 'js_', 'btn_', 'weui', 'img_loading',
}


def clean_html_bs4(html_content: str) -> str:
    """
    使用 BeautifulSoup 清洗 HTML（基础版）
    """
    if not BS4_AVAILABLE:
        return html_content
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 移除 script/style/meta/link 标签
    for tag in soup(['script', 'style', 'meta', 'link', 'noscript']):
        tag.decompose()
    
    # 移除所有 style/class/id 属性
    for tag in soup.find_all(True):
        tag.attrs = {k: v for k, v in tag.attrs.items() 
                     if k not in ['style', 'class', 'id', 'onclick', 'onload']}
    
    # 获取文本
    text = soup.get_text(separator=' ', strip=True)
    
    return text


def clean_html_trafilatura(html_content: str) -> str:
    """
    使用 trafilatura 提取正文（推荐，但需要完整 HTML）
    """
    if not TRAFILATURA_AVAILABLE:
        return clean_html_bs4(html_content)
    
    # 尝试提取正文
    text = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=False,
        include_images=False,
        include_links=False,
        favor_recall=True  # 宁可多提也不要漏
    )
    
    if text:
        return text
    
    # 失败时回退到 BeautifulSoup
    return clean_html_bs4(html_content)


def filter_css_noise(text: str) -> str:
    """
    过滤 CSS 相关噪音
    
    Args:
        text: 清洗后的文本
    
    Returns:
        过滤后的文本
    """
    if not text:
        return text
    
    # 转换为小写进行匹配
    text_lower = text.lower()
    
    # 标记需要删除的片段
    to_remove = []
    for keyword in CSS_KEYWORDS:
        # 找到所有匹配位置
        start = 0
        while True:
            idx = text_lower.find(keyword, start)
            if idx == -1:
                break
            
            # 提取前后上下文，判断是否是独立词
            before = text[idx-1:idx] if idx > 0 else ' '
            after = text[idx+len(keyword):idx+len(keyword)+1] if idx+len(keyword) < len(text) else ' '
            
            # 如果是独立词（前后是空格或标点），标记删除
            if before in ' \n\t，。！？；："\'（）【】《》' and after in ' \n\t，。！？；："\'（）【】《》:：=；\n':
                # 扩展到整个词/短语
                word_start = idx
                word_end = idx + len(keyword)
                
                # 向后扩展到空格或标点
                while word_end < len(text) and text[word_end] not in ' \n\t，。！？；："\'（）【】《》':
                    word_end += 1
                
                to_remove.append((word_start, word_end))
            
            start = idx + 1
    
    # 去重并排序（从后往前删，避免索引错乱）
    to_remove = sorted(set(to_remove), key=lambda x: x[0], reverse=True)
    
    # 删除标记的片段
    result = text
    for start, end in to_remove:
        result = result[:start] + result[end:]
    
    # 清理多余空白
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


def clean_wechat_article(html_content: str, method: str = 'auto') -> str:
    """
    微信公众号文章清洗主函数
    
    Args:
        html_content: 原始 HTML
        method: 清洗方法 ('auto', 'trafilatura', 'bs4')
    
    Returns:
        清洗后的纯文本
    """
    if not html_content:
        return ''
    
    # 第一步：HTML 结构提取
    if method == 'trafilatura' and TRAFILATURA_AVAILABLE:
        text = clean_html_trafilatura(html_content)
    elif method == 'bs4' and BS4_AVAILABLE:
        text = clean_html_bs4(html_content)
    else:
        # auto 模式：优先 trafilatura
        text = clean_html_trafilatura(html_content)
    
    # 第二步：CSS 噪音过滤
    text = filter_css_noise(text)
    
    # 第三步：终极正则绞肉机
    # 3.1 强制抹除所有类似 "font-family: xxx;", "margin: 0px;" 的键值对残骸
    text = re.sub(r'[a-zA-Z-]+:\s*[^;]+;', ' ', text)
    # 3.2 强制抹除孤立的乱码英文组合（长度小于3的大写字母组合，极大概率是残骸）
    text = re.sub(r'\b[A-Z]{1,2}\b', ' ', text)
    # 3.3 强制抹除 CSS 类名模式（短横线连接的小写字母）
    text = re.sub(r'\b[a-z]+-[a-z]+-[a-z]+\b', ' ', text)
    
    # 第四步：最终清理
    # 移除 URL
    text = re.sub(r'https?://\S+', '', text)
    # 移除 email
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    # 移除连续的特殊字符
    text = re.sub(r'[^\w\u4e00-\u9fa5]{3,}', ' ', text)
    # 清理多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# 便捷函数
def extract_text(html_content: str) -> str:
    """便捷函数：清洗微信公众号文章 HTML"""
    return clean_wechat_article(html_content, method='auto')
