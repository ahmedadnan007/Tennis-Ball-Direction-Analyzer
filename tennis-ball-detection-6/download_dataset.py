from roboflow import Roboflow

rf = Roboflow(api_key="ibeZ1CkOK8KPLUYUt3mX")

# 1000+ labeled tennis ball images download
project = rf.workspace("viren-dhanwani").project("tennis-ball-detection")
version = project.version(6)
dataset = version.download("yolov8")

print(f"✅ Dataset downloaded!")
print(f"📁 Location: {dataset.location}")