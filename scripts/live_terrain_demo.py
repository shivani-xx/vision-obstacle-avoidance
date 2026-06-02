# scripts\live_terrain_demo.py
import pybullet as p
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys, json
sys.path.insert(0, 'src')
from terrain_environments import create_terrain

# === Load your best trained model ===
device = torch.device('cpu')
CLASSES = ['flat_ground', 'uphill_slope', 'rough_terrain', 'hazard']
SPEEDS = {
    'flat_ground': 0.15,     # Full speed
    'uphill_slope': 0.08,    # Reduced speed
    'rough_terrain': 0.05,   # Slow
    'hazard': 0.0,           # STOP
}

# Load model (replace with your best architecture)
model = models.mobilenet_v3_small(weights=None)  # Or your chosen architecture
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
ckpt = torch.load('models\\best_terrain_model.pth', map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Inference transform (same as test transform)
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2(),
])


def predict_terrain(image_rgb):
    tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = output.argmax(1).item()
    return CLASSES[pred_idx], float(probs[pred_idx])


# Run demo on each terrain type (use unseen seed=999)
results = []
for terrain in CLASSES:
    client = p.connect(p.DIRECT)
    robot_id = create_terrain(terrain, seed=999)  # UNSEEN seed
    
    correct = 0
    total = 50  # 50 steps per terrain
    x = 3.0  # Start in the terrain zone
    
    for step in range(total):
        orient = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(robot_id, [x, 0, 0.5], orient)
        p.stepSimulation()
        
        # Capture and predict
        pos, quat = p.getBasePositionAndOrientation(robot_id)
        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)
        cam = [pos[0], pos[1], pos[2]+0.3]
        fwd = rot @ np.array([1,0,0])
        tgt = [cam[0]+fwd[0], cam[1]+fwd[1], cam[2]+fwd[2]]
        view = p.computeViewMatrix(cam, tgt, [0,0,1])
        proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100)
        _, _, rgba, _, _ = p.getCameraImage(224, 224, view, proj,
                                             renderer=p.ER_TINY_RENDERER)
        rgb = np.array(rgba, dtype=np.uint8).reshape(224,224,4)[:,:,:3]
        
        predicted, confidence = predict_terrain(rgb)
        speed = SPEEDS[predicted]
        if predicted == terrain:
            correct += 1
        x += speed
    
    accuracy = correct / total
    results.append({'terrain': terrain, 'accuracy': accuracy, 'correct': correct, 'total': total})
    print(f'  {terrain}: {accuracy:.1%} accuracy ({correct}/{total})')
    p.disconnect()

# Save and print summary
with open('results\\capstone_navigation.json', 'w') as f:
    json.dump(results, f, indent=2)
avg = sum(r['accuracy'] for r in results) / len(results)
print(f'\nOverall accuracy: {avg:.1%}')
