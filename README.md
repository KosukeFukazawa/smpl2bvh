# smpl2bvh
This repository contains an example script to convert from a SMPL model to a bvh file.

## Notification
This code is MIT licensed, but SMPL requires a separate license.  
Please see [SMPL official website](https://smpl.is.tue.mpg.de/).

## Requirements
You need to download SMPL models in `./data/smpl/smpl/`.  
You need to download [smplx](https://github.com/vchoutas/smplx) too.  
To install from PyPi simply run:
```
    pip install smplx[all]
```

## How to use?
After downloads all requirement, you can use smpl2bvh like this:
```python
    python smpl2bvh.py --gender MALE --poses data/poses.npz --fps 30 --output data/demo.bvh
```

`poses` is an npz file, which must contain `rotations` and `trans` as keys.  
`rotations` value is an array consisting of [fnum, 24, 3] and `trans` value is the root transition consisting of [fnum, 3]
(fnum means frame number).  
You can find bvh file as `output`.  
For more information, plese refer to `smpl2bvh.py`.  

## Reference
`bvh.py` and `quat.py` are based on [Motion Matching](https://github.com/orangeduck/Motion-Matching).