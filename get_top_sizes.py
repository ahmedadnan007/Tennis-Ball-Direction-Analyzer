import os
from pathlib import Path
base=Path(__file__).resolve().parent
entries=[p for p in base.iterdir() if p.name not in ('.git','.vscode')]
res={}
for e in entries:
    try:
        if e.is_file():
            res[e.name]={'type':'file','size_bytes':e.stat().st_size}
        else:
            total=0
            for root,dirs,files in os.walk(e):
                for f in files:
                    try:
                        total+=os.path.getsize(os.path.join(root,f))
                    except Exception:
                        pass
            res[e.name]={'type':'dir','size_bytes':total}
    except Exception as exc:
        res[e.name]={'error':str(exc)}

import json
print(json.dumps(res,indent=2))
