#!/usr/bin/env python3
"""Analyze pending blogs and generate TODO update."""
import json

# Load blog list
with open('/home/kelvin/dev/ml_posts/docs/data/blog_list.json', 'r', encoding='utf-8') as f:
    blogs = json.load(f)

# Count by status
completed = [b for b in blogs if b.get('status') == 'completed']
pending = [b for b in blogs if b.get('status') == 'pending']

print(f"Total blogs: {len(blogs)}")
print(f"Completed: {len(completed)}")
print(f"Pending: {len(pending)}")
print(f"Completion rate: {len(completed)/len(blogs)*100:.1f}%")
print()

# Sort pending by post_number (descending, most recent first)
pending_with_num = [b for b in pending if b.get('post_number') is not None]
pending_with_num.sort(key=lambda x: x['post_number'], reverse=True)

print("=" * 80)
print("PENDING BLOGS (sorted by post number, most recent first)")
print("=" * 80)
for b in pending_with_num[:50]:  # Top 50
    num = b.get('post_number', 'N/A')
    title = b.get('title', 'Untitled')
    date = b.get('date', 'N/A')
    print(f"{num:3} | {date:10} | {title}")

print()
print("=" * 80)
print("RECENTLY COMPLETED BLOGS (top 10)")
print("=" * 80)
completed_with_num = [b for b in completed if b.get('post_number') is not None]
completed_with_num.sort(key=lambda x: x['post_number'], reverse=True)
for b in completed_with_num[:10]:
    num = b.get('post_number', 'N/A')
    title = b.get('title', 'Untitled')
    date = b.get('date', 'N/A')
    print(f"{num:3} | {date:10} | {title}")
