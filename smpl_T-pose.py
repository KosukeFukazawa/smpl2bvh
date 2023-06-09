#get T-pose for smpl, in order to retarget pose from/to smpl.
import numpy as np
import argparse
import pickle
import smplx
from utils import bvh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/smpl/")
    parser.add_argument("--model_type", type=str, default="smpl", choices=["smpl", "smplx"])
    parser.add_argument("--gender", type=str, default="MALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", type=str, default="output.bvh")
    return parser.parse_args()


def smpl2bvh(model_path:str, output:str,model_type="smpl", gender="MALE", fps=60) -> None:
    """Save bvh file created by smpl parameters.

    Args:
        model_path (str): Path to smpl models.
        output (str): Where to save bvh.
        model_type (str, optional): I prepared "smpl" only. Defaults to "smpl".
        gender (str, optional): Gender Information. Defaults to "MALE".
        fps (int, optional): Frame per second. Defaults to 30.
    """
    
    names = [
        "Pelvis",
        "Left_hip",
        "Right_hip",
        "Spine1",
        "Left_knee",
        "Right_knee",
        "Spine2",
        "Left_ankle",
        "Right_ankle",
        "Spine3",
        "Left_foot",
        "Right_foot",
        "Neck",
        "Left_collar",
        "Right_collar",
        "Head",
        "Left_shoulder",
        "Right_shoulder",
        "Left_elbow",
        "Right_elbow",
        "Left_wrist",
        "Right_wrist",
        "Left_palm",
        "Right_palm",
    ]
    
    # I prepared smpl models only, 
    # but I will release for smplx models recently.
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
    
    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 100
    order = "zyx"
    
    rotation = np.zeros((1,52,3))
    positions = np.ones((1,24,3))
    positions[:,:,1:2] = positions[:,:,1:2]*100
    
    bvh_data ={
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": names,
        "order": order,
        "frametime": 1 / fps,
    }
    
    if not output.endswith(".bvh"):
        output = output + ".bvh"
    
    bvh.save(output, bvh_data)
    
  
if __name__ == "__main__":
    args = parse_args()
    
    smpl2bvh(model_path=args.model_path, model_type=args.model_type,gender=args.gender,fps=args.fps, output=args.output)
    
    print("finished!")
