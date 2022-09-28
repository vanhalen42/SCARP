# SCARP
SCARP: 3D Shape Completion in ARbitrary Poses for Improved Grasping

This repository contains the demo code for the paper submitted to ICRA, titled "*SCARP: 3D Shape Completion in ARbitrary Poses for Improved Grasping*"

## Pretrained Checkpoints
Pretrained checkpoints for the demo can be downloaded from [here](https://drive.google.com/drive/folders/137CSxW1AORyo2zG6BRFpd-UpJPLE86r2)

Move the checkpoints to the directory `checkpoints/`.

The directory structure should be:
```
└── checkpoints
    ├── plane.pt
    .
    .
    .

    └── car.pt
```

## Demo

To run a demo run the following command

```bash
CUDA_VISIBLE_DEVICES=0 python3 demo.py  \
--ckpt_load checkpoints/plane.pt
```