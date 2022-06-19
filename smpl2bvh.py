import torch
import numpy as np
import argparse
import smplx

from utils import bvh, quat

names = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_palm",
    "right_palm",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/smpl/")
    parser.add_argument("--model_type", type=str, default="smpl")
    parser.add_argument("--gender", type=str, default="MALE")
    parser.add_argument("--num_betas", type=int, default=10)
    parser.add_argument("--poses", type=str, default="data/demo.npz")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default="data/demo_smpl.bvh")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # I prepared smpl models only, but I will release for smplx models recently.
    model = smplx.create(model_path=args.model_path, 
                        model_type=args.model_type,
                        gender=args.gender, 
                        batch_size=1)
    
    parents = model.parents.detach().cpu().numpy()
    
    # You can define betas like this.(default betas are 0 at all.)
    rest = model(
        # betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    )
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:24,:]
    
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = np.zeros(3)
    
    poses = np.load(args.poses)
    rots = np.squeeze(poses["poses"], axis=0)
    trans = np.squeeze(poses["trans"], axis=0)
    
    order = "zyx"
    positions = offsets[None].repeat(len(rots), axis=0)
    positions[:,0] = trans
    angles = np.sqrt(np.sum(rots[...,0:1]**2 + rots[...,1:2]**2 + rots[...,2:3]**2, 
                            axis=-1, keepdims=True))
    axis = rots / angles
    angles = angles[...,0]
    rotations = np.degrees(quat.to_euler(quat.from_angle_axis(angles, axis), order=order))
    
    bvh_data ={
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": names,
        "order": order,
        "frametime": 1 / args.fps,
    }
    
    bvh.save(args.output, bvh_data)