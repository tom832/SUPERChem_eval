import pandas as pd
import numpy as np
import re
import base64
import os
import uuid as uuid_lib
from playwright.sync_api import sync_playwright
from typing import Dict, List, Tuple, Optional

class PatternLibrary:
    """
    所有正则集中管理，按匹配优先级排序
    """
    
    # === 结构化标签（最高优先级）===
    CHECKPOINT = r'<Checkpoint>(.*?)</Checkpoint>'
    MULTIMODAL = r'<MultiModal>(.*?)</MultiModal>'
    
    # === 公式（按长度优先：块级 > 行内）===
    FORMULA_BLOCK_DOLLAR = r'\$\$([\s\S]+?)\$\$'
    FORMULA_BLOCK_BRACKET = r'\\\[([\s\S]+?)\\\]'
    FORMULA_INLINE_DOLLAR = r'(?<!\$)\$(?!\$)((?:[^\$]|\\\$)+?)(?<!\$)\$(?!\$)'
    FORMULA_INLINE_PAREN = r'\\\(([\s\S]+?)\\\)'
    
    # === 轻量 Markdown（排除占位符）===
    # 粗体：**text** 或 __text__
    BOLD_DOUBLE_STAR = r'\*\*(.+?)\*\*'
    BOLD_DOUBLE_UNDER = r'(?<!%)__(?!%)(.*?)(?<!%)__(?!%)'
    
    # 斜体：*text* 或 _text_（排除占位符中的下划线）
    ITALIC_SINGLE_STAR = r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)'
    # 关键修复：斜体下划线正则排除 %% 占位符
    ITALIC_SINGLE_UNDER = r'(?<![%\w])_(?!_)([^_]+?)_(?![%\w])'

    # === 表格 ===
    # 匹配表格行：以 | 开头和结尾
    TABLE_ROW = r'^\|(.+)\|$'
    # 匹配分隔行：| --- | --- |
    TABLE_SEPARATOR = r'^\|[\s\-:|]+\|$'

class ContentProcessor:
    """
    统一的内容处理管道
    采用"提取-保护-处理-回填"策略，避免公式被 Markdown 破坏
    
    关键修复：使用 UUID 风格的安全占位符，避免被正则误匹配
    """
    
    # 占位符前后缀，使用不常见的字符组合
    PH_PREFIX = "%%"
    PH_SUFFIX = "%%"
    
    def __init__(self, images_map: Dict[str, bytes]):
        self.images_map = images_map
        self.storage: Dict[str, Dict] = {}  # {占位符: {"type": "F/MM/CP", "content": "..."}}
    
    def _generate_placeholder(self, type_prefix: str) -> str:
        """生成唯一的安全占位符"""
        unique_id = uuid_lib.uuid4().hex[:12]
        return f'{self.PH_PREFIX}{type_prefix}{unique_id}{self.PH_SUFFIX}'
    
    def process(self, text: str) -> str:
        """主入口：执行完整的处理管道"""
        if not text:
            return ""
        
        # Phase 1: 提取结构化标签
        text = self._extract_tags(text)
        
        # Phase 2: 提取公式
        text = self._extract_formulas(text)
        
        # Phase 3: 轻量文本格式化
        text = self._format_text(text)
        
        # Phase 4: 回填占位符
        html = self._restore_placeholders(text)
        
        return html

    # ==================== 辅助解析方法 ====================
    def _parse_bracket_content(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """
        反向解析括号内容
        从末尾找最后一组 (...) 作为第二部分
        剩余 [...] 作为第一部分
        
        返回: (方括号内容, 小括号内容) 或 (None, None)
        """
        content = content.strip()
        
        # 步骤1: 从末尾找 )
        if not content.endswith(')'):
            return None, None
        
        # 步骤2: 反向匹配找到对应的 (
        depth = 0
        paren_start = -1
        for i in range(len(content) - 1, -1, -1):
            if content[i] == ')':
                depth += 1
            elif content[i] == '(':
                depth -= 1
                if depth == 0:
                    paren_start = i
                    break
        
        if paren_start == -1:
            return None, None
        
        # 步骤3: 提取小括号内容
        paren_content = content[paren_start + 1 : -1]  # 去掉 ( 和 )
        
        # 步骤4: 剩余部分
        remaining = content[:paren_start].strip()
        
        # 步骤5: 去除 [ ] 或 ![ ]
        if remaining.startswith('!['):
            remaining = remaining[2:]
        elif remaining.startswith('['):
            remaining = remaining[1:]
        
        if remaining.endswith(']'):
            remaining = remaining[:-1]
        
        return remaining.strip(), paren_content.strip()
    
    # ==================== Phase 1: 提取结构化标签 ====================
    
    def _extract_tags(self, text: str) -> str:
        """提取 Checkpoint 和 MultiModal 标签"""
        
        # Checkpoint
        def cp_replacer(match):
            ph = self._generate_placeholder("CP")
            self.storage[ph] = {"type": "CP", "content": match.group(1)}
            return ph
        
        text = re.sub(
            PatternLibrary.CHECKPOINT, 
            cp_replacer, 
            text, 
            flags=re.IGNORECASE | re.DOTALL
        )
        
        # MultiModal
        def mm_replacer(match):
            ph = self._generate_placeholder("MM")
            self.storage[ph] = {"type": "MM", "content": match.group(1)}
            return ph
        
        text = re.sub(
            PatternLibrary.MULTIMODAL, 
            mm_replacer, 
            text, 
            flags=re.IGNORECASE | re.DOTALL
        )
        
        return text
    
    # ==================== Phase 2: 提取公式 ====================
    
    def _extract_formulas(self, text: str) -> str:
        """按优先级提取4种公式定界符"""
        
        # 1. 块级 $$...$$
        text = self._extract_formula_pattern(
            text,
            PatternLibrary.FORMULA_BLOCK_DOLLAR,
            delimiters=('$$', '$$'),
            formula_type="block"
        )
        
        # 2. 块级 \[...\]
        text = self._extract_formula_pattern(
            text,
            PatternLibrary.FORMULA_BLOCK_BRACKET,
            delimiters=(r'\[', r'\]'),
            formula_type="block"
        )
        
        # 3. 行内 $...$
        text = self._extract_formula_pattern(
            text,
            PatternLibrary.FORMULA_INLINE_DOLLAR,
            delimiters=('$', '$'),
            formula_type="inline"
        )
        
        # 4. 行内 \(...\)
        text = self._extract_formula_pattern(
            text,
            PatternLibrary.FORMULA_INLINE_PAREN,
            delimiters=(r'\(', r'\)'),
            formula_type="inline"
        )
        
        return text
    
    def _extract_formula_pattern(self, text: str, pattern: str,
                                  delimiters: Tuple[str, str],
                                  formula_type: str) -> str:
        """提取公式并保留原始定界符"""
        
        def replacer(match):
            ph = self._generate_placeholder("F")
            # 保留定界符，MathJax 需要它们
            content = delimiters[0] + match.group(1) + delimiters[1]
            self.storage[ph] = {
                "type": "F",
                "content": content,
                "formula_type": formula_type
            }
            return ph
        
        return re.sub(pattern, replacer, text)
    
    # ==================== Phase 3: 轻量文本格式化 ====================
    
    def _format_text(self, text: str) -> str:
        """轻量文本格式化：处理粗体、斜体、换行、表格"""
        lines = text.split('\n')
        formatted_lines = []
        
        # 表格状态
        in_table = False
        table_rows = []
        
        for line in lines:
            stripped = line.strip()
            
            # 检测表格行
            if stripped.startswith('|') and stripped.endswith('|'):
                if not in_table:
                    in_table = True
                    table_rows = []
                table_rows.append(stripped)
            else:
                # 非表格行，先处理之前积累的表格
                if in_table:
                    formatted_lines.append(self._render_table(table_rows))
                    in_table = False
                    table_rows = []
                
                # 处理普通行
                formatted_line = self._format_inline(line)
                
                if formatted_line.strip() == '':
                    formatted_lines.append('<div class="line-break"></div>')
                else:
                    formatted_lines.append(f'<p>{formatted_line}</p>')
        
        # 处理末尾的表格
        if in_table and table_rows:
            formatted_lines.append(self._render_table(table_rows))
        
        return '\n'.join(formatted_lines)

    def _render_table(self, rows: list) -> str:
        """将 Markdown 表格行转换为 HTML 表格"""
        if len(rows) < 2:
            # 不足两行（至少需要表头+分隔符），当作普通文本
            return '\n'.join(f'<p>{self._format_inline(r)}</p>' for r in rows)
        
        html_parts = ['<table class="md-table">']
        
        is_header = True
        for i, row in enumerate(rows):
            # 跳过分隔行 |---|---|
            if re.match(r'^\|[\s\-:|]+\|$', row):
                continue
            
            # 解析单元格
            cells = [c.strip() for c in row.strip('|').split('|')]
            
            if is_header:
                html_parts.append('<thead><tr>')
                for cell in cells:
                    cell_html = self._format_inline(cell)
                    html_parts.append(f'<th>{cell_html}</th>')
                html_parts.append('</tr></thead>')
                html_parts.append('<tbody>')
                is_header = False
            else:
                html_parts.append('<tr>')
                for cell in cells:
                    cell_html = self._format_inline(cell)
                    html_parts.append(f'<td>{cell_html}</td>')
                html_parts.append('</tr>')
        
        html_parts.append('</tbody></table>')
        
        return '\n'.join(html_parts)
    
    def _format_inline(self, text: str) -> str:
        """处理行内格式：粗体、斜体（保护占位符不被误匹配）"""
        
        # 粗体 **text**
        text = re.sub(PatternLibrary.BOLD_DOUBLE_STAR, r'<strong>\1</strong>', text)
        
        # 粗体 __text__（已在正则中排除占位符）
        text = re.sub(PatternLibrary.BOLD_DOUBLE_UNDER, r'<strong>\1</strong>', text)
        
        # 斜体 *text*
        text = re.sub(PatternLibrary.ITALIC_SINGLE_STAR, r'<em>\1</em>', text)
        
        # 斜体 _text_（已在正则中排除占位符）
        text = re.sub(PatternLibrary.ITALIC_SINGLE_UNDER, r'<em>\1</em>', text)
        
        return text
    
    # ==================== Phase 4: 回填占位符 ====================
    
    def _restore_placeholders(self, text: str) -> str:
        """回填所有占位符"""
        
        for placeholder, data in self.storage.items():
            ptype = data["type"]
            content = data["content"]
            
            if ptype == "F":
                # 公式：包裹 MathJax 容器
                formula_type = data.get("formula_type", "inline")
                css_class = "math-block" if formula_type == "block" else "math-inline"
                escaped = self._escape_html_for_formula(content)
                replacement = f'<span class="math-tex {css_class}">{escaped}</span>'
            
            elif ptype == "MM":
                # 图片
                replacement = self._render_multimodal(content)
            
            elif ptype == "CP":
                # Checkpoint
                replacement = self._render_checkpoint(content)
            
            else:
                replacement = content
            
            text = text.replace(placeholder, replacement)
        
        return text
    
    def _escape_html_for_formula(self, text: str) -> str:
        """转义公式中的 HTML 特殊字符"""
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        return text
    
    # ==================== 渲染辅助方法 ====================
    
    def _render_multimodal(self, content: str) -> str:
        """解析 MultiModal 内容，生成图片 HTML"""
        alt, url = self._parse_bracket_content(content)
        
        if alt is not None and url is not None:
            return self._get_image_html(url, alt)
        
        return f'<div class="error-box">[Invalid MultiModal: {self._escape_html_for_formula(content)}]</div>'    
    
    def _render_checkpoint(self, content: str) -> str:
        """解析 Checkpoint，递归处理内部内容"""
        desc, score = self._parse_bracket_content(content)
        
        if desc is not None and score is not None:
            # 递归处理描述内容
            sub_processor = ContentProcessor(self.images_map)
            desc_html = sub_processor.process(desc)
            
            return f'''
            <div class="checkpoint-card">
                <div class="checkpoint-header">
                    <span class="checkpoint-title">CHECKPOINT</span>
                    <span class="checkpoint-score">{score} PTS</span>
                </div>
                <div class="checkpoint-body">{desc_html}</div>
            </div>
            '''
        
        return f'<div class="error-box">[Invalid Checkpoint: {self._escape_html_for_formula(content)}]</div>'
    
    def _get_image_html(self, key: str, alt: str) -> str:
        """生成 Base64 图片 HTML"""
        img_bytes = self.images_map.get(key)
        
        if not img_bytes:
            return f'<div class="error-box">[Image Not Found: {key}]</div>'
        
        mime = self._detect_mime(img_bytes)
        b64 = base64.b64encode(img_bytes).decode('utf-8')
        safe_alt = self._escape_html_for_formula(alt)
        
        return f'''
        <figure class="image-figure">
            <img src="data:image/{mime};base64,{b64}" alt="{safe_alt}">
            <figcaption>{safe_alt}</figcaption>
        </figure>
        '''
    
    def _detect_mime(self, data: bytes) -> str:
        """检测图片 MIME 类型"""
        if data.startswith(b'\xff\xd8\xff'):
            return 'jpeg'
        elif data.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'png'
        elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
            return 'gif'
        elif data.startswith(b'RIFF') and len(data) > 12 and data[8:12] == b'WEBP':
            return 'webp'
        return 'png'

class ExamPDFGenerator:
    """PDF 生成器"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self._ensure_dir()
    
    def _ensure_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    @property
    def mathjax_config(self) -> str:
        """MathJax 配置脚本"""
        return '''
        <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: false,
                processEnvironments: true,
                macros: {
                    degree: "^{\\\\circ}",
                    degreeC: "^{\\\\circ}\\\\text{C}",
                    bold: ["\\\\mathbf{#1}", 1]
                }
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
                processHtmlClass: 'math-tex'
            },
            svg: {
                fontCache: 'global'
            },
            startup: {
                typeset: false
            }
        };
        </script>
        <script id="MathJax-script" async 
                src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
        </script>
        '''
    
    @property
    def css_styles(self) -> str:
        """页面 CSS 样式"""
        return '''
        @page {
            margin: 2.2cm 1.5cm 2cm 1.5cm;
            size: A4;
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: "Georgia", "Times New Roman", "SimSun", serif;
            line-height: 1.75;
            color: #333;
            font-size: 11pt;
            margin: 0;
            padding: 0;
        }
        
        h1, h2, h3 {
            font-family: "Arial", "Helvetica", "Microsoft YaHei", sans-serif;
            color: #1a1a1a;
            margin-top: 0;
        }
        
        /* 标题样式 - 简洁无边框 */
        h2 {
            font-size: 1.2em;
            font-weight: 600;
            font-style: bold;
            color: #1a1a1a;
            margin-bottom: 16px;
            margin-top: 0;
            padding-bottom: 8px;
            border-bottom: 1px solid #e5e5e5;
        }
        
        /* 隐藏默认头部（由页眉替代） */
        .header-info {
            display: none;
        }
        
        /* 题目区块 */
        .question-block {
            font-size: 1.05em;
            margin-bottom: 22px;
            line-height: 1.9;
        }
        
        .question-block p {
            margin: 0 0 14px 0;
            text-align: justify;
        }
        
        /* 段落和换行 */
        p {
            margin: 0 0 12px 0;
        }
        
        .line-break {
            height: 10px;
        }
        
        /* 选项列表 */
        .options-list {
            list-style: none;
            padding: 0;
            margin: 20px 0;
        }
        
        .option-item {
            margin-bottom: 10px;
            padding: 14px 18px;
            background: linear-gradient(to right, #fafbfc, #ffffff);
            border-radius: 8px;
            display: flex;
            align-items: flex-start;
            border: 1px solid #eaecef;
            transition: all 0.2s ease;
        }
        
        .option-item:nth-child(odd) {
            background: linear-gradient(to right, #f6f8fa, #ffffff);
        }
        
        .opt-key {
            font-weight: 700;
            min-width: 36px;
            color: #0366d6;
            font-size: 1.08em;
            font-family: "Arial", sans-serif;
        }
        
        .opt-val {
            flex: 1;
            padding-top: 1px;
        }
        
        .opt-val p {
            margin: 0;
        }
        
        /* 公式样式 */
        .math-tex {
            /* MathJax 处理 */
        }
        
        .math-block {
            display: block;
            text-align: center;
            margin: 24px 0;
            padding: 18px 0;
            background-color: #fafbfc;
            border-radius: 6px;
        }
        
        .math-inline {
            display: inline;
            padding: 0 2px;
        }
        
        /* 图片样式 */
        .image-figure {
            margin: 28px auto;
            text-align: center;
            max-width: 95%;
            page-break-inside: avoid;
        }
        
        .image-figure img {
            max-height: 300px;
            max-width: 80%;
            width: auto;
            height: auto;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
            border-radius: 10px;
            border: 1px solid #eaeaea;
        }
        
        .image-figure figcaption {
            font-size: 0.88em;
            color: #666;
            margin-top: 14px;
            font-family: "Arial", sans-serif;
        }
        
        /* Checkpoint 卡片 */
        .checkpoint-card {
            background-color: #ffffff;
            border-radius: 10px;
            margin: 24px 0;
            overflow: hidden;
            page-break-inside: avoid;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
            border: 1px solid #e1e4e8;
        }
        
        .checkpoint-header {
            background: linear-gradient(135deg, #28a745, #22863a);
            color: white;
            padding: 10px 18px;
            font-size: 0.88em;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
            letter-spacing: 0.3px;
        }
        
        .checkpoint-title {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .checkpoint-score {
            background: rgba(255, 255, 255, 0.2);
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 500;
        }
        
        .checkpoint-body {
            padding: 16px 18px;
            background: #fafdfb;
        }
        
        .checkpoint-body p {
            margin: 0;
        }
        
        /* 答案区块 */
        .answer-box {
            font-size: 1.1em;
            background: linear-gradient(135deg, #f0f7ff, #ffffff);
            padding: 20px 24px;
            margin-bottom: 28px;
            border-radius: 10px;
            border: 1px solid #d0e3f7;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .answer-label {
            color: #586069;
            font-weight: 600;
            font-family: "Arial", sans-serif;
        }
        
        .answer-value {
            color: #0366d6;
            font-weight: 700;
            font-size: 1.15em;
        }
        
        /* 解析区块 */
        .explanation-block {
            line-height: 1.9;
            text-align: justify;
        }
        
        .explanation-block p {
            margin: 0 0 14px 0;
        }
        
        /* 列表样式 */
        .content-list {
            margin: 14px 0;
            padding-left: 28px;
        }
        
        .content-list li {
            margin-bottom: 10px;
            line-height: 1.75;
        }

        /* Markdown 表格样式 */
        .md-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
            page-break-inside: avoid;
        }

        .md-table th,
        .md-table td {
            border: 1px solid #ddd;
            padding: 10px 14px;
            text-align: left;
        }

        .md-table th {
            background: linear-gradient(135deg, #f6f8fa, #eef1f4);
            font-weight: 600;
            color: #24292e;
        }

        .md-table tr:nth-child(even) {
            background-color: #fafbfc;
        }

        .md-table tr:hover {
            background-color: #f0f4f8;
        }

        /* HTML 原生表格也美化 */
        table:not(.md-table) {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        table:not(.md-table) th,
        table:not(.md-table) td {
            border: 1px solid #ddd;
            padding: 10px 14px;
        }

        table:not(.md-table) th {
            background-color: #f6f8fa;
            font-weight: 600;
        }
                
        /* 错误提示 */
        .error-box {
            background-color: #fffbdd;
            border: 1px solid #f9c513;
            color: #735c0f;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 14px 0;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 0.85em;
        }
        
        /* 分页 */
        .page-break {
            page-break-before: always;
            display: block;
            height: 1px;
        }
        
        /* 打印优化 */
        @media print {
            .checkpoint-card,
            .image-figure {
                page-break-inside: avoid;
            }
            
            body {
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }
        }
        '''
    
    def _get_language_labels(self, lang: str) -> Dict[str, str]:
        """获取语言标签配置"""
        labels = {
            'zh': {
                'question': '题目',
                'answer_title': '答案',
                'answer_label': '正确答案',
                'explanation': '详细解析'
            },
            'en': {
                'question': 'Question',
                'answer_title': 'Answer',
                'answer_label': 'Correct Answer',
                'explanation': 'Detailed Explanation'
            }
        }
        return labels.get(lang, labels['en'])
    
    def _merge_images(self, row: pd.Series) -> Dict[str, bytes]:
        """合并所有图片资源"""
        images = {}
        for col in ['question_images', 'options_images', 'explanation_images']:
            if col in row and row[col] and isinstance(row[col], dict):
                images.update(row[col])
        return images
    
    def _format_answer(self, answer) -> str:
        """格式化答案"""
        if isinstance(answer, (list, np.ndarray)):
            return ", ".join([str(x) for x in answer])
        return str(answer) if answer else ""
    
    def generate_html(self, row: pd.Series, lang: str) -> str:
        """生成完整的 HTML 页面"""
        labels = self._get_language_labels(lang)
        images = self._merge_images(row)
        
        # 处理题目
        processor_q = ContentProcessor(images)
        question_html = processor_q.process(row.get(f'question_{lang}', ''))
        
        # 处理选项
        options_html = ""
        options_data = row.get(f'options_{lang}')
        if options_data and isinstance(options_data, dict):
            options_html = '<ul class="options-list">'
            for key in sorted(options_data.keys()):
                value = options_data[key]
                if value:
                    processor_opt = ContentProcessor(images)
                    value_html = processor_opt.process(str(value))
                    options_html += f'''
                    <li class="option-item">
                        <span class="opt-key">{key}.</span>
                        <div class="opt-val">{value_html}</div>
                    </li>
                    '''
            options_html += '</ul>'
        
        # 处理答案
        answer_str = self._format_answer(row.get(f'answer_{lang}'))
        
        # 处理解析
        processor_exp = ContentProcessor(images)
        explanation_html = processor_exp.process(row.get(f'explanation_{lang}', ''))
        
        # 注意：移除了 header-info div
        return f'''
        <!DOCTYPE html>
        <html lang="{lang}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SUPERChem - {row.get('uuid', 'Unknown')}</title>
            <style>{self.css_styles}</style>
            {self.mathjax_config}
        </head>
        <body>
            <!-- Page 1: Question -->
            <section class="section">
                <h2>{labels['question']}</h2>
                <div class="question-block">{question_html}</div>
                {options_html}
            </section>
            
            <div class="page-break"></div>
            
            <!-- Page 2: Answer -->
            <section class="section">
                <h2>{labels['answer_title']}</h2>
                <div class="answer-box">
                    <span class="answer-label">{labels['answer_label']}:</span>
                    <span class="answer-value">{answer_str}</span>
                </div>
                
                <h2>{labels['explanation']}</h2>
                <div class="explanation-block">
                    {explanation_html}
                </div>
            </section>
        </body>
        </html>
        '''
    
    def process_dataframe(self, df: pd.DataFrame, languages: List[str] = None):
        """处理整个 DataFrame 生成 PDF"""
        if languages is None:
            languages = ['zh', 'en']
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            
            total = len(df)
            for idx, row in df.iterrows():
                uuid = row.get('uuid', f'unknown_{idx}')
                canary = row.get('canary', 'CONFIDENTIAL')
                
                for lang in languages:
                    pdf_path = os.path.join(self.output_dir, f"{uuid}_{lang}.pdf")
                    
                    if os.path.exists(pdf_path):
                        print(f"[SKIP] {pdf_path}")
                        continue
                    
                    try:
                        html_content = self.generate_html(row, lang)
                        
                        page = browser.new_page()
                        page.set_content(html_content, wait_until="networkidle")
                        
                        # MathJax 渲染
                        page.evaluate('''
                            () => {
                                if (window.MathJax && window.MathJax.typesetPromise) {
                                    return MathJax.typesetPromise();
                                }
                                return Promise.resolve();
                            }
                        ''')
                        
                        page.wait_for_timeout(1500)
                        
                        # 页眉：左侧UUID，右侧SUPERChem带链接，底部横线
                        header_html = f'''
                        <div style="width: 100%; padding: 0 1.5cm; font-family: Arial, Helvetica, sans-serif; font-size: 10pt;">
                            <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 8px; border-bottom: 1px solid #ccc;">
                                <span style="color: #666; font-family: Consolas, Monaco, monospace; letter-spacing: 0.3px;">
                                    UUID: {uuid}
                                </span>
                                <a href="https://arxiv.org/abs/2512.01274" 
                                style="color: #0366d6; text-decoration: none; font-weight: 600; letter-spacing: 0.5px;">
                                    SUPERChem
                                </a>
                            </div>
                        </div>
                        '''
                        
                        # 页脚：保密信息和页码
                        footer_html = f'''
                        <div style="width: 100%; text-align: center; font-family: Arial, Helvetica, sans-serif; font-size: 9px; color: #999; padding-top: 8px;">
                            <span style="letter-spacing: 0.3px;">{canary}</span>
                            <span style="margin-left: 20px;">
                                Page <span class="pageNumber"></span> of <span class="totalPages"></span>
                            </span>
                        </div>
                        '''
                        
                        page.pdf(
                            path=pdf_path,
                            format="A4",
                            print_background=True,
                            display_header_footer=True,
                            header_template=header_html,
                            footer_template=footer_html,
                            margin={
                                "top": "2.2cm",  # 增加顶部边距给页眉留空间
                                "bottom": "2cm",
                                "left": "1.5cm",
                                "right": "1.5cm"
                            }
                        )
                        
                        page.close()
                        print(f"[OK] ({idx + 1}/{total}) {pdf_path}")
                        
                    except Exception as e:
                        print(f"[ERROR] {uuid}_{lang}: {e}")
                        import traceback
                        traceback.print_exc()
            
            browser.close()
            print(f"\n[DONE] Output: {self.output_dir}")

# ==================== 主程序 ====================

if __name__ == "__main__":
    parquet_path = 'SUPERChem-500.parquet'
    
    if not os.path.exists(parquet_path):
        print(f"[ERROR] File not found: {parquet_path}")
        exit(1)
    
    df = pd.read_parquet(parquet_path)
    print(f"[INFO] Loaded {len(df)} records")
    
    generator = ExamPDFGenerator(output_dir="./pdf")
    generator.process_dataframe(df, languages=['zh', 'en'])