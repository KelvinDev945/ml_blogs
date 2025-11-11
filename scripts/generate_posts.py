#!/usr/bin/env python3
"""
Generate HTML posts from markdown files in blogs_raw/

Usage:
    python generate_posts.py                  # Generate all posts
    python generate_posts.py --incremental    # Only regenerate modified posts
    python generate_posts.py --dry-run        # Preview what will be generated
    python generate_posts.py <file.md>        # Generate specific file
"""

import re
import json
import markdown
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from jinja2 import Template, Environment, FileSystemLoader

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
BLOGS_RAW_DIR = PROJECT_ROOT / "blogs_raw"
POSTS_DIR = PROJECT_ROOT / "docs" / "posts"
TEMPLATE_DIR = PROJECT_ROOT / "templates"
TEMPLATE_FILE = TEMPLATE_DIR / "post_template.html"
BLOG_LIST_JSON = PROJECT_ROOT / "docs" / "data" / "blog_list.json"
CACHE_FILE = PROJECT_ROOT / ".build_cache.json"

# Markdown extensions with enhanced support
MD_EXTENSIONS = [
    'extra',           # Tables, footnotes, etc.
    'codehilite',      # Code highlighting
    'toc',             # Table of contents
    'tables',          # GFM tables
    'fenced_code',     # Fenced code blocks
    # 'nl2br',         # Newline to <br> - REMOVED: breaks LaTeX formulas
    'attr_list',       # Add HTML attributes to elements
    'md_in_html',      # Allow Markdown inside HTML
    'def_list',        # Definition lists
    'footnotes',       # Footnotes support
    'sane_lists',      # Better list handling
]

# Extension configs
MD_EXTENSION_CONFIGS = {
    'codehilite': {
        'linenums': False,
        'css_class': 'highlight',
    },
    'toc': {
        'permalink': True,
        'permalink_class': 'toc-link',
        'toc_depth': 3,
    },
}


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
                key = key.strip()
                value = value.strip()

                # Handle boolean values
                if value.lower() in ('true', 'false'):
                    frontmatter[key] = value.lower() == 'true'
                else:
                    frontmatter[key] = value

    return frontmatter, body


def get_file_hash(file_path):
    """Calculate MD5 hash of file content"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_build_cache():
    """Load build cache to track file changes"""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_build_cache(cache):
    """Save build cache"""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def clean_latex_br_tags(html):
    """Remove <br /> tags that break LaTeX formulas

    This function removes <br /> tags that appear within LaTeX delimiters.
    These tags are inserted by markdown when it sees trailing spaces,
    but they break LaTeX rendering.
    """
    import re

    # Pattern to match LaTeX equations with <br /> inside
    # Handle inline $...$ and display $$...$$ and \begin{...}\end{...}
    patterns = [
        # \begin{equation}...\end{equation} with <br /> tags
        (r'(\\begin\{[^}]+\}.*?\\end\{[^}]+\})', re.DOTALL),
        # $$...$$ with <br /> tags
        (r'(\$\$.*?\$\$)', re.DOTALL),
        # $...$ with <br /> tags (but not $$)
        (r'(?<!\$)(\$[^\$]+?\$)(?!\$)', re.DOTALL),
    ]

    for pattern, flags in patterns:
        def remove_br(match):
            latex_block = match.group(1)
            # Remove all <br />, <br/>, and <br> tags within LaTeX
            cleaned = re.sub(r'<br\s*/?\s*>', '', latex_block)
            return cleaned

        html = re.sub(pattern, remove_br, html, flags=flags)

    return html


def convert_md_to_html(md_content):
    """Convert markdown to HTML with TOC support"""
    md = markdown.Markdown(
        extensions=MD_EXTENSIONS,
        extension_configs=MD_EXTENSION_CONFIGS
    )
    html = md.convert(md_content)

    # Clean up <br /> tags that break LaTeX formulas
    html = clean_latex_br_tags(html)

    # Get TOC if available
    toc = getattr(md, 'toc', '')

    return html, toc


def generate_post_html(md_file, post_number=None, dry_run=False):
    """Generate HTML post from a markdown file

    Args:
        md_file: Path to markdown file
        post_number: Sequential number for this post (based on date)
        dry_run: If True, don't write files, just report what would be done

    Returns:
        dict with post metadata
    """
    if not dry_run:
        print(f"Processing: {md_file.name}")
    else:
        print(f"Would process: {md_file.name}")

    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse frontmatter and body
    frontmatter, body = parse_frontmatter(content)

    # Extract metadata
    title = frontmatter.get('title', md_file.stem)
    slug = frontmatter.get('slug', md_file.stem)
    date = frontmatter.get('date', '')
    source_url = frontmatter.get('source', '')
    status = frontmatter.get('status', 'pending')
    tags_str = frontmatter.get('tags', '')
    tags = [t.strip() for t in tags_str.split(',')] if tags_str else []
    tags_reviewed = frontmatter.get('tags_reviewed', False)

    # Convert markdown to HTML
    html_content, toc = convert_md_to_html(body)

    # Generate description from first paragraph
    clean_text = re.sub(r'<[^>]+>', '', html_content)
    description = clean_text[:200].strip()
    if len(clean_text) > 200:
        description += "..."

    # Prepare template context
    context = {
        'title': title,
        'description': description,
        'date': date,
        'content': html_content,
        'source_url': source_url,
        'tags': tags,
        'toc': toc,
        'post_number': post_number,
        'slug': slug,
    }

    if not dry_run:
        # Use Jinja2 for proper templating
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template = env.get_template('post_template.html')

        try:
            html = template.render(**context)
        except Exception as e:
            print(f"  Warning: Template rendering issue, falling back to simple replacement: {e}")
            # Fallback to simple replacement
            with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
                html = f.read()

            # Simple replacements
            html = html.replace('{{ title }}', title)
            html = html.replace('{{ description }}', description)
            html = html.replace('{{ date }}', date)
            html = html.replace('{{ content }}', html_content)
            html = html.replace('{{ source_url }}', source_url)

            # Remove Jinja2 conditionals (simple approach)
            if not source_url:
                html = re.sub(r'\{% if source_url %\}.*?\{% endif %\}', '', html, flags=re.DOTALL)
            if not tags:
                html = re.sub(r'\{% if tags %\}.*?\{% endif %\}', '', html, flags=re.DOTALL)

        # Save HTML file
        output_file = POSTS_DIR / f"{slug}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"  ✓ Generated: {output_file.name}")

    # Return metadata
    return {
        'slug': slug,
        'title': title,
        'description': description,
        'date': date,
        'source': source_url,
        'tags': tags,
        'status': status,
        'tags_reviewed': tags_reviewed,
        'post_number': post_number,
    }


def get_post_files_with_dates():
    """Get all markdown files and extract their dates for sorting

    Returns:
        List of (md_file, date_str) tuples sorted by date (oldest first)
    """
    md_files = list(BLOGS_RAW_DIR.glob("*.md"))
    files_with_dates = []

    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            frontmatter, _ = parse_frontmatter(content)
            date_str = frontmatter.get('date', '9999-12-31')  # Unknown dates go last
            files_with_dates.append((md_file, date_str))
        except Exception as e:
            print(f"Warning: Could not read date from {md_file.name}: {e}")
            files_with_dates.append((md_file, '9999-12-31'))

    # Sort by date (oldest first)
    files_with_dates.sort(key=lambda x: x[1])

    return files_with_dates


def generate_all_posts(incremental=False, dry_run=False):
    """Generate HTML for all markdown files in blogs_raw/

    Args:
        incremental: Only regenerate files that have changed
        dry_run: Preview what would be generated without actually writing files
    """
    POSTS_DIR.mkdir(exist_ok=True, parents=True)
    (PROJECT_ROOT / "docs" / "data").mkdir(exist_ok=True, parents=True)

    # Get files sorted by date
    files_with_dates = get_post_files_with_dates()

    if not files_with_dates:
        print("No markdown files found in blogs_raw/")
        return

    # Load build cache for incremental builds
    build_cache = load_build_cache() if incremental else {}

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Generating {len(files_with_dates)} blog posts...")
    print(f"Mode: {'Incremental' if incremental else 'Full rebuild'}\n")

    # First pass: collect all metadata
    posts_metadata = []
    for post_number, (md_file, date_str) in enumerate(files_with_dates, start=1):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            frontmatter, _ = parse_frontmatter(content)

            posts_metadata.append({
                'md_file': md_file,
                'post_number': post_number,
                'slug': frontmatter.get('slug', md_file.stem),
                'title': frontmatter.get('title', md_file.stem),
                'date': frontmatter.get('date', ''),
            })
        except Exception as e:
            print(f"Warning: Could not read metadata from {md_file.name}: {e}")

    # Second pass: generate HTML with prev/next info
    processed_count = 0
    skipped_count = 0
    final_metadata = []

    for i, post_meta in enumerate(posts_metadata):
        md_file = post_meta['md_file']
        post_number = post_meta['post_number']

        try:
            # Check if file needs processing (for incremental mode)
            file_hash = get_file_hash(md_file)
            cache_key = str(md_file)

            # Get prev/next post info
            prev_post = posts_metadata[i - 1] if i > 0 else None
            next_post = posts_metadata[i + 1] if i < len(posts_metadata) - 1 else None

            # Remove md_file from prev/next before passing to template
            if prev_post:
                prev_post = {k: v for k, v in prev_post.items() if k != 'md_file'}
            if next_post:
                next_post = {k: v for k, v in next_post.items() if k != 'md_file'}

            if incremental and cache_key in build_cache:
                if build_cache[cache_key] == file_hash:
                    # File hasn't changed, but regenerate to update prev/next links
                    if not dry_run:
                        print(f"Updating navigation: {md_file.name}")

                    metadata = generate_post_html_with_nav(
                        md_file, post_number, prev_post, next_post, dry_run
                    )
                    final_metadata.append(metadata)
                    skipped_count += 1
                    continue

            # Generate post
            metadata = generate_post_html_with_nav(
                md_file, post_number, prev_post, next_post, dry_run
            )
            final_metadata.append(metadata)

            # Update cache
            if not dry_run:
                build_cache[cache_key] = file_hash

            processed_count += 1

        except Exception as e:
            print(f"  ✗ Error processing {md_file.name}: {e}")
            import traceback
            traceback.print_exc()

    # Save build cache
    if not dry_run and incremental:
        save_build_cache(build_cache)

    # Generate blog_list.json
    if not dry_run:
        # Sort by date for the JSON (newest first for main page display)
        final_metadata.sort(key=lambda x: x.get('date', '1900-01-01'), reverse=True)

        # Ensure all posts have post_number (assign based on original order)
        for i, post in enumerate(final_metadata, start=1):
            if 'post_number' not in post or post['post_number'] is None:
                post['post_number'] = i

        with open(BLOG_LIST_JSON, 'w', encoding='utf-8') as f:
            json.dump(final_metadata, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Generated blog_list.json with {len(final_metadata)} posts")

    # Summary
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Processed: {processed_count}")
    if incremental:
        print(f"  Skipped (unchanged): {skipped_count}")
    print(f"  Total: {len(files_with_dates)}")
    if not dry_run:
        print(f"  Output: {POSTS_DIR}")
        print(f"  Blog list: {BLOG_LIST_JSON}")


def generate_post_html_with_nav(md_file, post_number, prev_post, next_post, dry_run=False):
    """Wrapper that adds prev/next navigation to generate_post_html"""
    if not dry_run:
        print(f"Processing: {md_file.name}")
    else:
        print(f"Would process: {md_file.name}")

    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse frontmatter and body
    frontmatter, body = parse_frontmatter(content)

    # Extract metadata
    title = frontmatter.get('title', md_file.stem)
    slug = frontmatter.get('slug', md_file.stem)
    date = frontmatter.get('date', '')
    source_url = frontmatter.get('source', '')
    status = frontmatter.get('status', 'pending')
    tags_str = frontmatter.get('tags', '')
    tags = [t.strip() for t in tags_str.split(',')] if tags_str else []
    tags_reviewed = frontmatter.get('tags_reviewed', False)

    # Convert markdown to HTML
    html_content, toc = convert_md_to_html(body)

    # Generate description from first paragraph
    clean_text = re.sub(r'<[^>]+>', '', html_content)
    description = clean_text[:200].strip()
    if len(clean_text) > 200:
        description += "..."

    # Prepare template context (including prev/next)
    context = {
        'title': title,
        'description': description,
        'date': date,
        'content': html_content,
        'source_url': source_url,
        'tags': tags,
        'toc': toc,
        'post_number': post_number,
        'slug': slug,
        'prev_post': prev_post,
        'next_post': next_post,
    }

    if not dry_run:
        # Use Jinja2 for proper templating
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template = env.get_template('post_template.html')

        try:
            html = template.render(**context)
        except Exception as e:
            print(f"  Warning: Template rendering issue: {e}")
            # Fallback to simple replacement (won't have prev/next)
            with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
                html = f.read()

            html = html.replace('{{ title }}', title)
            html = html.replace('{{ description }}', description)
            html = html.replace('{{ date }}', date)
            html = html.replace('{{ content }}', html_content)

        # Save HTML file
        output_file = POSTS_DIR / f"{slug}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"  ✓ Generated: {output_file.name}")

    # Return metadata
    return {
        'slug': slug,
        'title': title,
        'description': description,
        'date': date,
        'source': source_url,
        'tags': tags,
        'status': status,
        'tags_reviewed': tags_reviewed,
        'post_number': post_number,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate HTML posts from Markdown files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_posts.py                      # Full rebuild
  python generate_posts.py --incremental        # Only rebuild changed files
  python generate_posts.py --dry-run            # Preview changes
  python generate_posts.py file.md              # Generate single file
  python generate_posts.py --incremental --dry-run  # Preview incremental
        """
    )

    parser.add_argument(
        'file',
        nargs='?',
        help='Specific markdown file to process'
    )
    parser.add_argument(
        '--incremental', '-i',
        action='store_true',
        help='Only regenerate files that have changed'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be generated without actually writing files'
    )

    args = parser.parse_args()

    if args.file:
        # Generate specific file
        md_file = Path(args.file)
        if md_file.exists():
            # Get post number for this file
            files_with_dates = get_post_files_with_dates()
            post_number = None
            for i, (f, _) in enumerate(files_with_dates, start=1):
                if f == md_file:
                    post_number = i
                    break

            metadata = generate_post_html(md_file, post_number=post_number, dry_run=args.dry_run)
            print(f"\nGenerated post #{post_number}: {metadata['title']}")
        else:
            print(f"Error: File not found: {md_file}")
            exit(1)
    else:
        # Generate all posts
        generate_all_posts(incremental=args.incremental, dry_run=args.dry_run)
