#!/usr/bin/env python3
import os, re, markdown
from pathlib import Path
from datetime import datetime

SOURCE_DIR = Path(__file__).parent / "blogs_raw"
OUTPUT_DIR = Path(__file__).parent / "blog_site"
POSTS_DIR = OUTPUT_DIR / "posts"

CSS_STYLE = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5;
}
.container { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
header { border-bottom: 2px solid #e0e0e0; padding-bottom: 20px; margin-bottom: 30px; }
h1 { color: #2c3e50; margin-bottom: 10px; font-size: 2em; }
.meta { color: #666; font-size: 0.9em; margin-bottom: 20px; }
.content { margin-top: 30px; overflow-wrap: break-word; }
.content h2 { color: #34495e; margin-top: 30px; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
.content p { margin-bottom: 15px; }
.content pre { background: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; border-left: 3px solid #3498db; margin: 15px 0; }
.content code { background: #f8f8f8; padding: 2px 5px; border-radius: 3px; font-family: monospace; font-size: 0.9em; }
.content blockquote { border-left: 4px solid #3498db; padding-left: 20px; margin: 20px 0; color: #555; font-style: italic; }
.content table { border-collapse: collapse; width: 100%; margin: 20px 0; }
.content th, .content td { border: 1px solid #ddd; padding: 10px; text-align: left; }
.back-link { display: inline-block; margin-bottom: 20px; color: #3498db; text-decoration: none; font-weight: 500; }
"""

MATHJAX_CONFIG = r"""
<script>
    window.MathJax = {
        tex: {
            inlineMath: [['$', '$'], ['\(', '\)']],
            displayMath: [['$$', '$$'], ['\[', '\]']],
            processEscapes: true
        }
    };
</script>
<script id="MathJax-script" async src="https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

def generate_post(md_file: Path):
    content = md_file.read_text(encoding='utf-8')
    
    # 1. 移除 Front Matter (使用非捕获组和更简单的匹配)
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            content = parts[2]
    
    # 2. 提取并移除标题
    title = md_file.stem.replace('_', ' ')
    lines = content.split('\n')
    clean_lines = []
    found_h1 = False
    for line in lines:
        if not found_h1 and line.strip().startswith('# '):
            title = line.strip()[2:].strip()
            found_h1 = True
            continue
        clean_lines.append(line)
    content = '\n'.join(clean_lines)
    
    # 3. 保护数学公式 (使用原始字符串 regex)
    math_map = {}
    def math_repl(m):
        key = f"---MATHPLACEHOLDER{len(math_map)}---"
        math_map[key] = m.group(0)
        return key
    
    # 匹配顺序很重要：先匹配大的块，再匹配行内
    # 块级: $$, \[, \begin...
    # 行内: $
    patterns = [
        r'\$\$.*?\$\$',
        r'\\\\[.*?\\\\\]',
        r'\\begin\{.*?\}.*?\\[^e]end\{.*?\}',
        r'(?<!\$)\\(?!\$)\\(?<!\$)\\(?!\$)'
    ]
    combined_pattern = '|'.join(patterns)
    content = re.sub(combined_pattern, math_repl, content, flags=re.DOTALL)
    
    # 4. Markdown 转 HTML
    html_body = markdown.markdown(content, extensions=['extra', 'codehilite', 'toc'])
    
    # 5. 还原数学公式 (直接替换)
    for key, original in math_map.items():
        html_body = html_body.replace(key, original)
    
    # 6. 组装
    mtime = datetime.fromtimestamp(md_file.stat().st_mtime).strftime('%Y-%m-%d')
    size = f"{md_file.stat().st_size / 1024:.1f} KB"
    
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{CSS_STYLE}</style>
    {MATHJAX_CONFIG}
</head>
<body>
    <div class="container">
        <a href="../index.html" class="back-link">← 返回首页</a>
        <header>
            <h1>{title}</h1>
            <div class="meta">📅 最后更新: {mtime} | 📄 大小: {size}</div>
        </header>
        <div class="content">
            {html_body}
        </div>
    </div>
</body>
</html>"""
    return full_html, title

def main():
    print("🚀 正在重新生成站点...")
    if not OUTPUT_DIR.exists(): OUTPUT_DIR.mkdir()
    if not POSTS_DIR.exists(): POSTS_DIR.mkdir()
    
    md_files = list(SOURCE_DIR.glob("*.md"))
    posts_info = []
    
    for i, md in enumerate(md_files):
        try:
            html, title = generate_post(md)
            (POSTS_DIR / (md.stem + ".html")).write_text(html, encoding='utf-8')
            posts_info.append({
                'f': md.stem + ".html", 
                't': title, 
                'd': datetime.fromtimestamp(md.stat().st_mtime).strftime('%Y-%m-%d'), 
                's': f"{md.stat().st_size/1024:.1f} KB"
            })
            if (i+1) % 50 == 0: print(f"  已完成 {i+1}/{len(md_files)} 篇")
        except Exception as e:
            print(f"  ❌ 处理 {md.name} 时出错: {e}")
            
    # 生成索引页
    posts_info.sort(key=lambda x: x['d'], reverse=True)
    li_html = "".join([f'<li class="post-item"><a href="posts/{p["f"]}">{p["t"]}</a><div class="post-meta">📅 {p["d"]} | 📄 {p["s"]}</div></li>' for p in posts_info])
    
    index_style = """
    body { font-family: sans-serif; background: #764ba2; padding: 20px; }
    .container { max-width: 900px; margin: 0 auto; background: white; border-radius: 10px; padding: 40px; }
    .post-item { background: #f8f9fa; padding: 15px; margin-bottom: 10px; border-radius: 5px; list-style: none; }
    .post-item a { color: #2c3e50; text-decoration: none; font-weight: bold; }
    .post-meta { color: #666; font-size: 0.8em; }
    """
    
    index_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>机器学习博客</title><style>{index_style}</style></head>
<body>
    <div class="container">
        <h1>🤖 机器学习博客</h1>
        <p>文章总数: {len(posts_info)}</p>
        <ul>{li_html}</ul>
    </div>
</body>
</html>"""
    (OUTPUT_DIR / "index.html").write_text(index_html, encoding='utf-8')
    print(f"✅ 完成！共生成 {len(posts_info)} 篇文章。")

if __name__ == "__main__":
    main()