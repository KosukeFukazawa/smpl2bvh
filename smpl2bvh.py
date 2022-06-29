import torch
import numpy as np
import argparse
import pickle
import smplx

from utils import bvh, quat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/smpl/")
    parser.add_argument("--model_type", type=str, default="smpl", choices=["smpl", "smplx"])
    parser.add_argument("--gender", type=str, default="MALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--num_betas", type=int, default=10, choices=[10, 300])
    parser.add_argument("--poses", type=str, default="data/demo.npz")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default="data/demo_smpl.bvh")
    return parser.parse_args()

def smpl2bvh(model_path:str, poses:str, output:str, 
             model_type="smpl", gender="MALE", num_betas=10, fps=30)->None:
    """Save bvh file created by smpl parameters.

    Args:
        model_path (str): Path to smpl models.
        poses (str): Path to npz or pkl file.
        output (str): Where to save bvh.
        model_type (str, optional): I prepared "smpl" only. Defaults to "smpl".
        gender (str, optional): Gender Information. Defaults to "MALE".
        num_betas (int, optional): How many pca parameters to use in SMPL. Defaults to 10.
        fps (int, optional): Frame per second. Defaults to 30.
    """
    
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
    
    # I prepared smpl models only, but I will release for smplx models recently.
    model = smplx.create(model_path=model_path, 
                        model_type=model_type,
                        gender=gender, 
                        batch_size=1)
    
    parents = model.parents.detach().cpu().numpy()
    
    # You can define betas like this.(default betas are 0 at all.)
    rest = model(
        # betas = torch.randn([1, num_betas], dtype=torch.float32)
    )
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:24,:]
    
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = np.zeros(3)
    
    scaling = None
    
    # Pose setting.
    if poses.endswith(".npz"):
        poses = np.load(poses)
        rots = np.squeeze(poses["poses"], axis=0) # (N, 24, 3)
        trans = np.squeeze(poses["trans"], axis=0) # (N, 3)

    elif poses.endswith(".pkl"):
        with open(poses, "rb") as f:
            poses = pickle.load(f)
            rots = poses["smpl_poses"] # (N, 72)
            rots = rots.reshape(rots.shape[0], -1, 3) # (N, 24, 3)
            scaling = poses['smpl_scaling']  # (1,)
            trans = poses['smpl_trans']  # (N, 3)
    
    else:
        raise Exception("This file type is not supported!")
    
    if scaling is not None:
        trans /= scaling
    
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
        "frametime": 1 / fps,
    }
    
    bvh.save(output, bvh_data)

if __name__ == "__main__":
    args = parse_args()
    
    smpl2bvh(model_path=args.model_path, model_type=args.model_type, gender=args.gender,
             poses=args.poses, num_betas=args.num_betas, fps=args.fps, output=args.output)