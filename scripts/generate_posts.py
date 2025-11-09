#!/usr/bin/env python3
"""
Generate HTML posts from markdown files in blogs_raw/
"""

import re
import json
import markdown
from pathlib import Path
from jinja2 import Template

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
BLOGS_RAW_DIR = PROJECT_ROOT / "blogs_raw"
POSTS_DIR = PROJECT_ROOT / "docs" / "posts"
TEMPLATE_FILE = PROJECT_ROOT / "templates" / "post_template.html"
BLOG_LIST_JSON = PROJECT_ROOT / "docs" / "data" / "blog_list.json"

# Markdown extensions
MD_EXTENSIONS = [
    'extra',
    'codehilite',
    'toc',
    'tables',
    'fenced_code',
    'nl2br'
]


def parse_frontmatter(content):
    """Parse YAML-like frontmatter from markdown"""
    frontmatter = {}
    body = content

    # Match frontmatter between --- markers
    match = re.match(r'^---\s*\n(.*?\n)---\s*\n(.*)', content, re.DOTALL)
    if match:
        fm_text = match.group(1)
        body = match.group(2)

        # Parse simple key: value pairs
        for line in fm_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                frontmatter[key.strip()] = value.strip()

    return frontmatter, body


def convert_md_to_html(md_content):
    """Convert markdown to HTML"""
    md = markdown.Markdown(extensions=MD_EXTENSIONS)
    html = md.convert(md_content)
    return html


def generate_post_html(md_file):
    """Generate HTML post from a markdown file"""
    print(f"Processing: {md_file.name}")

    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse frontmatter and body
    frontmatter, body = parse_frontmatter(content)

    # Convert markdown to HTML
    html_content = convert_md_to_html(body)

    # Read template
    with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
        template_content = f.read()

    # Simple template replacement (Jinja2-style)
    title = frontmatter.get('title', md_file.stem)
    slug = frontmatter.get('slug', md_file.stem)
    date = frontmatter.get('date', '')
    source_url = frontmatter.get('source', '')
    tags_str = frontmatter.get('tags', '')
    tags = [t.strip() for t in tags_str.split(',')] if tags_str else []

    # Generate description from first paragraph
    description = re.sub(r'<[^>]+>', '', html_content)[:200] + "..."

    # Replace template variables
    html = template_content
    html = html.replace('{{ title }}', title)
    html = html.replace('{{ description }}', description)
    html = html.replace('{{ date }}', date)
    html = html.replace('{{ content }}', html_content)

    # Handle conditional source_url
    if source_url:
        source_html = f'''
                <span class="ms-3">
                    <i class="fas fa-link"></i>
                    <a href="{source_url}" target="_blank">原文链接</a>
                </span>'''
        html = html.replace('{% if source_url %}', '')
        html = html.replace('{{ source_url }}', source_url)
        html = html.replace('{% endif %}', '')
    else:
        # Remove the source_url block
        html = re.sub(r'\{% if source_url %\}.*?\{% endif %\}', '', html, flags=re.DOTALL)

    # Handle tags
    if tags:
        tags_html = '\n'.join([f'                <span class="tag"><i class="fas fa-tag"></i> {tag}</span>'
                               for tag in tags])
        html = html.replace('{% if tags %}', '')
        html = html.replace('{% for tag in tags %}', '')
        html = html.replace('                <span class="tag"><i class="fas fa-tag"></i> {{ tag }}</span>',
                           tags_html)
        html = html.replace('{% endfor %}', '')
        html = html.replace('{% endif %}', '', 1)  # First occurrence
    else:
        # Remove tags block
        html = re.sub(r'\{% if tags %\}.*?\{% endif %\}', '', html, flags=re.DOTALL, count=1)

    # Save HTML file
    output_file = POSTS_DIR / f"{slug}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  ✓ Generated: {output_file.name}")

    return slug


def update_blog_status(slug, status='completed'):
    """Update blog status in blog_list.json"""
    try:
        with open(BLOG_LIST_JSON, 'r', encoding='utf-8') as f:
            blogs = json.load(f)

        for blog in blogs:
            if blog['slug'] == slug:
                blog['status'] = status
                break

        with open(BLOG_LIST_JSON, 'w', encoding='utf-8') as f:
            json.dump(blogs, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"  Warning: Could not update blog status: {e}")


def generate_all_posts(update_status=True):
    """Generate HTML for all markdown files in blogs_raw/"""
    POSTS_DIR.mkdir(exist_ok=True)

    md_files = list(BLOGS_RAW_DIR.glob("*.md"))

    if not md_files:
        print("No markdown files found in blogs_raw/")
        return

    print(f"\nGenerating {len(md_files)} blog posts...\n")

    for md_file in md_files:
        try:
            slug = generate_post_html(md_file)
            if update_status:
                update_blog_status(slug, 'completed')
        except Exception as e:
            print(f"  ✗ Error processing {md_file.name}: {e}")

    print(f"\n✓ Generated {len(md_files)} HTML posts in {POSTS_DIR}")


if __name__ == "__main__":
    import sys

    # Check if specific file provided
    if len(sys.argv) > 1:
        md_file = Path(sys.argv[1])
        if md_file.exists():
            generate_post_html(md_file)
        else:
            print(f"File not found: {md_file}")
    else:
        # Generate all posts
        generate_all_posts()
