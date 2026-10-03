"""Check relative markdown links and fail on byte-identical duplicate files. Usage: python tools/check_links.py (run from the repo root).
Known placeholders in the project-readme template are ignored."""
import sys
import re,os,sys
bad=[]
for root,d,fs in os.walk('.'):
    if '.git' in root or '_research' in root: continue
    for f in fs:
        if not f.endswith('.md'): continue
        p=os.path.join(root,f); t=open(p,errors='ignore').read()
        t=re.sub(r'```.*?```','',t,flags=re.S)
        for m in re.finditer(r'\]\(([^)\s]+)\)',t):
            l=m.group(1)
            if re.match(r'(https?:|mailto:|#)',l): continue
            l=l.split('#')[0]
            if l and not os.path.exists(os.path.normpath(os.path.join(root,l))): bad.append((p,l))
import hashlib,collections
seen=collections.defaultdict(list)
for root,d,fs in os.walk('.'):
    if '.git' in root.split(os.sep): continue
    for f in fs:
        fp=os.path.join(root,f)
        if os.path.getsize(fp)>200 and 'config-pack' not in fp:
            seen[hashlib.sha256(open(fp,'rb').read()).hexdigest()].append(fp)
dups=[v for v in seen.values() if len(v)>1]
for v in dups: print('DUPLICATE FILES:',*v)
IGNORE={('./01-hackathon-playbook/templates/project-readme.md','docs/screenshot.png'),('./01-hackathon-playbook/templates/project-readme.md','LICENSE')}
bad=[b for b in bad if b not in IGNORE]
for b in bad: print(*b)
print(len(bad),'broken links,',len(dups),'duplicate groups')
sys.exit(1 if (bad or dups) else 0)
