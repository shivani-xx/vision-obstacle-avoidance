import pybullet as p
import pybullet_data
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
import time
import json

from src.environments import create_environment

# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------

device = torch.device('cpu')

ACTIONS = [
    'go_straight',
    'turn_left',
    'turn_right',
    'stop'
]

# Load MobileNetV3-Small
model = models.mobilenet_v3_small(weights=None)

model.classifier[-1] = nn.Linear(
    model.classifier[-1].in_features,
    4
)

# Load trained weights
checkpoint = torch.load(
    'models\\mobilenet_v3_small_navigation.pth',
    map_location=device
)

# Handle both checkpoint formats
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

print('Model loaded successfully!')

# ---------------------------------------------------
# Inference transform
# SAME preprocessing used during training
# ---------------------------------------------------

inference_transform = A.Compose([
    A.Resize(224, 224),

    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    ToTensorV2(),
])

# ---------------------------------------------------
# Prediction function
# ---------------------------------------------------

def predict_action(model, image_rgb, transform):

    """
    Predict robot action from camera image.
    """

    augmented = transform(image=image_rgb)

    tensor = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(output, dim=1)[0]

        predicted_idx = output.argmax(dim=1).item()

    return ACTIONS[predicted_idx], probabilities.numpy()

# ---------------------------------------------------
# Live navigation deployment
# ---------------------------------------------------

def run_live_navigation(
    difficulty='medium',
    num_steps=300,
    seed=99
):

    """
    Deploy trained CNN to navigate autonomously.
    """

    # GUI mode so you can SEE the robot
    physics_client = p.connect(p.GUI)

    # Create environment
    create_environment(
        difficulty=difficulty,
        seed=seed
    )

    # Load robot
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    robot_id = p.loadURDF(
        'r2d2.urdf',
        [0, 0, 0.5]
    )

    x, y, angle = 0.0, 0.0, 0.0

    speed = 0.12
    turn_rate = 0.12

    collision_count = 0

    action_log = []

    print(f'\nLive navigation: {difficulty}')
    print(f'Using unseen environment seed={seed}\n')

    for step in range(num_steps):

        # ---------------------------------------------------
        # Position robot
        # ---------------------------------------------------

        orient = p.getQuaternionFromEuler([0, 0, angle])

        p.resetBasePositionAndOrientation(
            robot_id,
            [x, y, 0.5],
            orient
        )

        p.stepSimulation()

        # ---------------------------------------------------
        # Capture camera image
        # ---------------------------------------------------

        pos, quat = p.getBasePositionAndOrientation(robot_id)

        rot = np.array(
            p.getMatrixFromQuaternion(quat)
        ).reshape(3, 3)

        cam = [
            pos[0],
            pos[1],
            pos[2] + 0.3
        ]

        fwd = rot @ np.array([1, 0, 0])

        tgt = [
            cam[0] + fwd[0],
            cam[1] + fwd[1],
            cam[2] + fwd[2]
        ]

        view = p.computeViewMatrix(
            cam,
            tgt,
            [0, 0, 1]
        )

        proj = p.computeProjectionMatrixFOV(
            60,
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
        ).reshape(224, 224, 4)[:, :, :3]

        # ---------------------------------------------------
        # CNN predicts action
        # ---------------------------------------------------

        action, probs = predict_action(
            model,
            rgb,
            inference_transform
        )

        action_log.append({
            'step': step,
            'action': action,
            'x': round(x, 2),
            'y': round(y, 2)
        })

        # ---------------------------------------------------
        # Execute action
        # ---------------------------------------------------

        if action == 'go_straight':

            x += speed * math.cos(angle)
            y += speed * math.sin(angle)

        elif action == 'turn_left':

            angle += turn_rate

        elif action == 'turn_right':

            angle -= turn_rate

        # ---------------------------------------------------
        # Collision detection
        # ---------------------------------------------------

        contacts = p.getContactPoints(bodyA=robot_id)

        obstacle_contacts = [
            c for c in contacts if c[2] != 0
        ]

        if obstacle_contacts:
            collision_count += 1

        # ---------------------------------------------------
        # Reset if robot escapes environment
        # ---------------------------------------------------

        if x > 22 or x < -3 or abs(y) > 4:

            x, y, angle = 0.0, 0.0, 0.0

        # ---------------------------------------------------
        # Logging
        # ---------------------------------------------------

        if step % 50 == 0:

            conf = max(probs)

            print(
                f'Step {step:>3}: '
                f'action={action:<12} '
                f'pos=({x:.1f},{y:.1f}) '
                f'conf={conf:.1%}'
            )

        time.sleep(1 / 60)

    p.disconnect()

    # ---------------------------------------------------
    # Results summary
    # ---------------------------------------------------

    action_counts = {}

    for entry in action_log:

        action_counts[entry['action']] = (
            action_counts.get(entry['action'], 0) + 1
        )

    collision_rate = collision_count / num_steps * 100

    max_x = max(entry['x'] for entry in action_log)

    print(f'\n{"="*50}')
    print(f'NAVIGATION RESULTS ({difficulty})')
    print(f'{"="*50}')

    print(f'Steps completed: {num_steps}')
    print(f'Collisions: {collision_count} ({collision_rate:.1f}%)')
    print(f'Max distance: {max_x:.1f}m')

    print(f'Action breakdown:')

    for action, count in sorted(action_counts.items()):

        print(
            f'  {action}: '
            f'{count} ({count/num_steps:.1%})'
        )

    return {
        'difficulty': difficulty,
        'steps': num_steps,
        'collisions': collision_count,
        'collision_rate': round(collision_rate, 1),
        'max_distance': round(max_x, 1),
        'actions': action_counts
    }

# ---------------------------------------------------
# Run all difficulty levels
# ---------------------------------------------------

nav_results = []

for diff in ['easy', 'medium', 'hard']:

    result = run_live_navigation(
        difficulty=diff,
        num_steps=300,
        seed=99
    )

    nav_results.append(result)

# ---------------------------------------------------
# Save results
# ---------------------------------------------------

with open('results\\navigation_results.json', 'w') as f:

    json.dump(nav_results, f, indent=2)

print('\nNavigation results saved!')