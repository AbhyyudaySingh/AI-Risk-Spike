import json

with open("artifacts/manifests/feature_manifest.json", "r") as f:
    manifest = json.load(f)

for col in manifest.get("excluded", {}):
    manifest["excluded"][col]["reason"] = "Not included in v1 controlled baseline feature set."

with open("artifacts/manifests/feature_manifest.json", "w") as f:
    json.dump(manifest, f, indent=4)
print("Updated feature_manifest.json")
