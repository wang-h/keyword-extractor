"""
HTML 清洗工具 - 专门针对微信公众号文章优化

问题：原始 HTML 包含大量 CSS 样式代码，干扰实体提取
解决：多层过滤，保留纯文本内容；避免误删合法缩写（AI、AR）与中文「作者：」句式
"""
import re
from typing import FrozenSet, Optional, Set

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


# CSS 相关关键词黑名单（用于 filter_css_noise）
CSS_KEYWORDS = {
    "helvetica",
    "arial",
    "microsoft yahei",
    "microsoft yahei ui",
    "hiragino sans gb",
    "pingfang sc",
    "noto sans",
    "sans-serif",
    "serif",
    "monospace",
    "courier new",
    "times new roman",
    "wenquanyi micro hei",
    "stheiti",
    "stheiti light",
    "padding",
    "margin",
    "font-size",
    "line-height",
    "color:",
    "background",
    "border",
    "display:",
    "width:",
    "height:",
    "text-align",
    "vertical-align",
    "overflow",
    "position:",
    "z-index",
    "float:",
    "clear:",
    "opacity",
    "visibility",
    "mp;",
    "data-",
    "aria-",
    "role=",
    "tabindex",
    "draggable",
    "contenteditable",
    "spellcheck",
    "translate",
    "dir=",
    "rich_media",
    "js_",
    "btn_",
    "weui",
    "img_loading",
}

# 完整 CSS 属性名（长词优先），避免用「color」前缀误匹配 colorful 等英文词
_COMMON_CSS_PROPS = [
    "-webkit-line-clamp",
    "-ms-text-size-adjust",
    "background-color",
    "background-image",
    "background-repeat",
    "border-radius",
    "border-bottom",
    "border-color",
    "letter-spacing",
    "margin-bottom",
    "margin-right",
    "margin-left",
    "padding-bottom",
    "padding-right",
    "padding-left",
    "text-decoration",
    "vertical-align",
    "pointer-events",
    "justify-content",
    "align-items",
    "flex-direction",
    "grid-template",
    "overflow-x",
    "overflow-y",
    "white-space",
    "word-break",
    "word-wrap",
    "line-height",
    "font-family",
    "font-weight",
    "font-size",
    "font-style",
    "box-sizing",
    "list-style",
    "object-fit",
    "column-gap",
    "row-gap",
    "background",
    "visibility",
    "transform",
    "transition",
    "animation",
    "position",
    "overflow",
    "outline",
    "padding",
    "margin",
    "border",
    "height",
    "width",
    "display",
    "opacity",
    "z-index",
    "float",
    "clear",
    "color",
    "content",
    "cursor",
    "flex",
    "grid",
    "gap",
    "top",
    "left",
    "right",
    "bottom",
    "max-width",
    "min-width",
    "max-height",
    "min-height",
    "text-align",
]

_COMMON_CSS_PROPS.sort(key=len, reverse=True)
_CSS_DECLARATION_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(p) for p in _COMMON_CSS_PROPS)
    + r")\s*:\s*[^;\n]{1,500};",
    re.IGNORECASE,
)

# 极长的小写连字符串（多见于 class 名），4 段以上且每段至少 2 字符
_CSSISH_SLUG_RE = re.compile(
    r"\b[a-z]{2,}(?:-[a-z]{2,}){3,}\b",
    re.IGNORECASE,
)


def clean_html_bs4(html_content: str) -> str:
    """使用 BeautifulSoup 清洗 HTML（基础版）"""
    if not BS4_AVAILABLE:
        return html_content

    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["script", "style", "meta", "link", "noscript"]):
        tag.decompose()

    for tag in soup.find_all(True):
        tag.attrs = {
            k: v
            for k, v in tag.attrs.items()
            if k not in ["style", "class", "id", "onclick", "onload"]
        }

    text = soup.get_text(separator=" ", strip=True)
    return text


def clean_html_trafilatura(html_content: str) -> str:
    """使用 trafilatura 提取正文（推荐，但需要完整 HTML）"""
    if not TRAFILATURA_AVAILABLE:
        return clean_html_bs4(html_content)

    text = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=False,
        include_images=False,
        include_links=False,
        favor_recall=True,
    )

    if text:
        return text

    return clean_html_bs4(html_content)


def _should_preserve_span(
    text: str,
    start: int,
    end: int,
    preserve_keywords: Optional[Set[str]],
) -> bool:
    """若片段与 preserve_keywords 中任一词重叠则保留（大小写不敏感）。"""
    if not preserve_keywords:
        return False
    span_lower = text[start:end].lower()
    for kw in preserve_keywords:
        if not kw:
            continue
        kl = kw.lower()
        if kl in span_lower or span_lower in kl:
            return True
    return False


def filter_css_noise(
    text: str,
    preserve_keywords: Optional[Set[str]] = None,
) -> str:
    """
    过滤 CSS 相关噪音。

    Args:
        text: 清洗后的文本
        preserve_keywords: 绝不删除的短语集合（与删除区间有重叠则跳过该删除）
    """
    if not text:
        return text

    text_lower = text.lower()
    to_remove = []
    for keyword in CSS_KEYWORDS:
        start = 0
        while True:
            idx = text_lower.find(keyword, start)
            if idx == -1:
                break

            before = text[idx - 1 : idx] if idx > 0 else " "
            after = (
                text[idx + len(keyword) : idx + len(keyword) + 1]
                if idx + len(keyword) < len(text)
                else " "
            )

            if before in " \n\t，。！？；：\"'（）【】《》" and after in " \n\t，。！？；：\"'（）【】《》:：=；\n":
                word_start = idx
                word_end = idx + len(keyword)
                while word_end < len(text) and text[word_end] not in " \n\t，。！？；：\"'（）【】《》":
                    word_end += 1
                if not _should_preserve_span(text, word_start, word_end, preserve_keywords):
                    to_remove.append((word_start, word_end))

            start = idx + 1

    to_remove = sorted(set(to_remove), key=lambda x: x[0], reverse=True)
    result = text
    for s, e in to_remove:
        result = result[:s] + result[e:]

    result = re.sub(r"\s+", " ", result).strip()
    return result


def strip_known_css_declarations(text: str) -> str:
    """仅删除白名单 CSS 属性形式的 `key: value;` 片段。"""
    return _CSS_DECLARATION_RE.sub(" ", text)


def clean_wechat_article(
    html_content: str,
    method: str = "auto",
    preserve_keywords: Optional[Set[str]] = None,
) -> str:
    """
    微信公众号文章清洗主函数

    Args:
        html_content: 原始 HTML 或纯文本
        method: 清洗方法 ('auto', 'trafilatura', 'bs4')
        preserve_keywords: 需要保留的术语/实体短语，避免被 CSS 噪音规则误伤

    Returns:
        清洗后的纯文本
    """
    if not html_content:
        return ""

    frozen_preserve: Optional[FrozenSet[str]] = (
        frozenset(preserve_keywords) if preserve_keywords else None
    )
    _pk: Optional[Set[str]] = set(frozen_preserve) if frozen_preserve else None

    if method == "trafilatura" and TRAFILATURA_AVAILABLE:
        text = clean_html_trafilatura(html_content)
    elif method == "bs4" and BS4_AVAILABLE:
        text = clean_html_bs4(html_content)
    else:
        text = clean_html_trafilatura(html_content)

    text = filter_css_noise(text, preserve_keywords=_pk)

    # 仅删除白名单 CSS 声明，不再使用贪婪的 [a-zA-Z-]+:\s*[^;]+;
    text = strip_known_css_declarations(text)

    # 已移除：\b[A-Z]{1,2}\b —— 会误删 AI、AR、VR、5G 等关键词

    # 长串 slug 更像残留 class 名；短横词如 two-word-name 不误伤
    text = _CSSISH_SLUG_RE.sub(" ", text)

    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fa5]{3,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_text(html_content: str) -> str:
    """便捷函数：清洗微信公众号文章 HTML"""
    return clean_wechat_article(html_content, method="auto")
