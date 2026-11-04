#!/usr/bin/env python3
"""Verify supplemental file hashes and portable guide links without imports/network."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/meeting_20261104'
class Links(HTMLParser):
 def __init__(self): super().__init__(); self.links=[]; self.ids=set()
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if 'id' in d:self.ids.add(d['id'])
  if tag=='a' and 'href' in d:self.links.append(d['href'])
  if tag in ('img','script') and 'src' in d:self.links.append(d['src'])
errors=[]
manifest=json.loads((DOC/'SUPPLEMENT_MANIFEST.json').read_text())
for item in manifest['files']:
 p=ROOT/item['path']
 if not p.is_file():errors.append('missing: '+item['path']);continue
 b=p.read_bytes()
 if len(b)!=item['bytes'] or hashlib.sha256(b).hexdigest()!=item['sha256']:errors.append('changed: '+item['path'])
local_links=0
sources=list(DOC.glob('*.md'))+[DOC/'visual_guide.html']
for p in sources:
 ids=None
 if p.suffix=='.html':
  parser=Links();parser.feed(p.read_text()); links=parser.links;ids=parser.ids
 else:links=re.findall(r'\]\(([^)]+)\)',p.read_text())
 for link in links:
  url=urlsplit(link)
  if url.scheme or url.netloc:continue
  if url.path:
   local_links+=1
   if not (p.parent/unquote(url.path)).exists():errors.append(p.name+' missing link: '+link)
  elif url.fragment and ids is not None and url.fragment not in ids:errors.append(p.name+' missing anchor: '+link)
for p in (ROOT/'tools/receiver').glob('*.py'):
 try:compile(p.read_text(),str(p),'exec')
 except (SyntaxError,UnicodeError) as e:errors.append(str(e))
result={'status':'FAIL' if errors else 'PASS','files_verified':len(manifest['files']),'local_links_checked':local_links,'errors':errors}
print(json.dumps(result,ensure_ascii=False,separators=(',',':')))
sys.exit(bool(errors))
