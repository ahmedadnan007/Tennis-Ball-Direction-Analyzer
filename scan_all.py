import os,sys,json
from pathlib import Path
base=Path(__file__).resolve().parent
candidates=[]
large_files=[]
for root, dirs, files in os.walk(base):
    # skip .git
    if '.git' in root.split(os.sep):
        continue
    for d in dirs:
        dn=d.lower()
        full=Path(root)/d
        if dn.startswith('.venv') or dn=='venv' or 'env'==dn or dn=='env' or dn=='archive_unused' or dn=='__pycache__' or dn=='node_modules':
            # compute size
            total=0
            for r,ds,fs in os.walk(full):
                for f in fs:
                    try:
                        total+=os.path.getsize(os.path.join(r,f))
                    except Exception:
                        pass
            candidates.append({'path':str(full.relative_to(base)), 'type':'dir', 'size_bytes':total, 'reason':'virtualenv/archive/pycache or node_modules'})
    for f in files:
        fn=f.lower()
        full=Path(root)/f
        try:
            size=os.path.getsize(full)
        except Exception:
            size=0
        # large model files
        if fn.endswith('.pt') or fn.endswith('.pth'):
            candidates.append({'path':str(full.relative_to(base)), 'type':'file','size_bytes':size,'reason':'model weights (.pt/.pth)'} )
        if fn.endswith('.ipynb'):
            candidates.append({'path':str(full.relative_to(base)), 'type':'file','size_bytes':size,'reason':'jupyter notebook'} )
        if fn.endswith('.mp4') and 'outputs' not in str(root).lower():
            # raw videos outside outputs
            candidates.append({'path':str(full.relative_to(base)), 'type':'file','size_bytes':size,'reason':'video file outside outputs'})
        if size>50*1024*1024:
            large_files.append({'path':str(full.relative_to(base)), 'size_bytes':size})
# dedupe
seen=set()
unique=[]
for c in candidates:
    if c['path'] in seen: continue
    seen.add(c['path']); unique.append(c)
out={'base':str(base), 'candidates':unique, 'large_files':large_files}
print(json.dumps(out, indent=2))
