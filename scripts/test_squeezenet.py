import torch

checkpoint = torch.load(
    "models/squeezenet_224.pth",
    map_location="cpu"
)

print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nKeys:")
    print(checkpoint.keys())