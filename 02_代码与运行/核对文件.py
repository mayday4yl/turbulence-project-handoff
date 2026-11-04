#!/usr/bin/env python3
"""Verify preserved originals, current copies and reader links without network or models."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / '03_归档资料'
MANIFEST = ROOT / '02_代码与运行/资料维护/资料清单.json'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.add(attrs['id'])
        if tag == 'a' and attrs.get('href'):
            self.links.append(attrs['href'])
        if tag in ('img', 'script') and attrs.get('src'):
            self.links.append(attrs['src'])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads(MANIFEST.read_text())
    errors, original_checks = [], []
    for group in ('archive_files', 'current_files'):
        for item in manifest[group]:
            path = ROOT / item['path']
            if not path.is_file() or path.is_symlink():
                errors.append('缺失或不是普通文件：' + item['path'])
            elif digest(path) != item['sha256']:
                errors.append('文件身份不符：' + item['path'])
    for item in manifest['current_code_copies']:
        current, original = ROOT / item['copy'], ROOT / item['original']
        if not current.is_file() or not original.is_file() or current.read_bytes() != original.read_bytes():
            errors.append('当前代码副本与原件不一致：' + item['copy'])
    for script in ('tools/verify_repository.py', 'tools/receiver/verify_supplement.py'):
        result = subprocess.run([sys.executable, str(ARCHIVE / script)], capture_output=True, text=True)
        try:
            record = json.loads(result.stdout)
        except ValueError:
            record = {'status': 'FAIL', 'output': result.stdout, 'error': result.stderr}
        original_checks.append({'script': script, 'result': record})
        if result.returncode or record.get('status') != 'PASS':
            errors.append('原始归档校验失败：' + script)
    sources = [ROOT / 'README.md', ROOT / '开始阅读.html', ROOT / '02_代码与运行/README.md', ARCHIVE / '归档目录.md']
    sources += list((ROOT / '01_交接说明').glob('*.md'))
    link_count = 0
    for path in sources:
        ids = None
        if path.suffix == '.html':
            parser = Links()
            parser.feed(path.read_text())
            links, ids = parser.links, parser.ids
        else:
            links = re.findall(r'\]\(([^)]+)\)', path.read_text())
        for link in links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            link_count += 1
            if url.path and not (path.parent / unquote(url.path)).exists():
                errors.append(str(path.relative_to(ROOT)) + ' 的链接缺失：' + link)
            elif not url.path and url.fragment and ids is not None and url.fragment not in ids:
                errors.append('页面章节不存在：' + link)
    for path in [ROOT / '02_代码与运行/运行示例.py', ROOT / '02_代码与运行/核对文件.py', ROOT / '02_代码与运行/资料维护/生成阅读页.py']:
        try:
            compile(path.read_text(), str(path), 'exec')
        except SyntaxError as exc:
            errors.append(str(exc))
    report = {'status': 'FAIL' if errors else 'PASS', 'preserved_original_files': len(manifest['archive_files']),
              'current_files': len(manifest['current_files']), 'identical_current_code_copies': len(manifest['current_code_copies']),
              'local_links_checked': link_count, 'archive_checks': original_checks, 'errors': errors}
    print(json.dumps(report, ensure_ascii=False, separators=(',', ':')))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
