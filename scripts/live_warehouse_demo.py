import sys
sys.path.append('.')

import pybullet as p
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

from src.warehouse_environments import create_environment


CLASS_NAMES = {
    0: 'open_aisle',
    1: 'narrow_aisle',
    2: 'pick_station',
    3: 'blocked_path',
    4: 'cross_aisle_junction'
}

ACTION_MAP = {
    'open_aisle': 'MOVE_FAST',
    'narrow_aisle': 'MOVE_SLOW',
    'pick_station': 'STOP_FOR_PICK',
    'blocked_path': 'EMERGENCY_STOP',
    'cross_aisle_junction': 'CAUTIOUS_FORWARD'
}

DEVICE = torch.device("cpu")


model = models.squeezenet1_1(weights=None)

model.classifier[1] = nn.Conv2d(
    512,
    5,
    kernel_size=1
)

state_dict = torch.load(
    "models/squeezenet_224.pth",
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model.eval()

print("SqueezeNet loaded successfully")


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


def get_robot_camera(robot_id):

    pos, orient = p.getBasePositionAndOrientation(robot_id)

    rot = np.array(
        p.getMatrixFromQuaternion(orient)
    ).reshape(3,3)

    fwd = rot @ np.array([1,0,0])

    cam_pos = [
        pos[0] - 0.3*fwd[0],
        pos[1] - 0.3*fwd[1],
        pos[2] + 0.8
    ]

    target = [
        pos[0] + fwd[0]*3.0,
        pos[1] + fwd[1]*3.0,
        pos[2] + 0.2
    ]

    view = p.computeViewMatrix(
        cam_pos,
        target,
        [0,0,1]
    )

    proj = p.computeProjectionMatrixFOV(
        70,
        1.0,
        0.1,
        100
    )

    _, _, rgba, _, _ = p.getCameraImage(
        224,
        224,
        view,
        proj,
        renderer=p.ER_TINY_RENDERER
    )

    rgb = np.array(
        rgba,
        dtype=np.uint8
    ).reshape(224,224,4)[:,:,:3]

    return rgb


def predict_scene(image):

    image = Image.fromarray(image)

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(tensor)

        pred_idx = outputs.argmax(1).item()

    return pred_idx


for class_idx in range(5):

    print("\n" + "="*50)

    p.connect(p.GUI)

    robot_id = create_environment(
        class_idx=class_idx,
        seed=999
    )

    p.stepSimulation()

    image = get_robot_camera(robot_id)

    pred_idx = predict_scene(image)

    pred_class = CLASS_NAMES[pred_idx]

    action = ACTION_MAP[pred_class]

    print("Ground Truth :", CLASS_NAMES[class_idx])
    print("Prediction   :", pred_class)
    print("Robot Action :", action)

    input("\nPress ENTER for next scene...")

    p.disconnect()

print("\nDemo complete.")