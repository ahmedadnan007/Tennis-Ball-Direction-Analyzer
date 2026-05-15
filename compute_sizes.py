import os,sys,json
base=os.path.dirname(__file__)
items=['.venv312','archive_unused','yolov8m.pt','runs']
res={}
for it in items:
    path=os.path.join(base,it)
    if os.path.exists(path):
        if os.path.isfile(path):
            res[it]={'type':'file','size_bytes':os.path.getsize(path)}
        else:
            total=0
            for root,dirs,files in os.walk(path):
                for f in files:
                    try:
                        total+=os.path.getsize(os.path.join(root,f))
                    except Exception:
                        pass
            res[it]={'type':'dir','size_bytes':total}
    else:
        res[it]={'exists':False}
print(json.dumps(res,indent=2))
