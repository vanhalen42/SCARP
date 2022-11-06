# SCARP
SCARP: 3D Shape Completion in ARbitrary Poses for Improved Grasping

This repository contains the demo code for the paper submitted to ICRA, titled "*SCARP: 3D Shape Completion in ARbitrary Poses for Improved Grasping*"
### Creating the Anaconda/Miniconda environment
Make sure you have Anaconda or Miniconda installed before you proceed to load this environment.
```
conda env create -f environment.yml
conda activate SCARP
```
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
--class_choice plane \
--ckpt_load checkpoints/plane.pt
```
The outputs are stored in the `demo_data/<class_choice>` directory

# Acknowledgement

Some parts of the code are insipired and borrowed from [ConDor](https://github.com/brown-ivl/ConDor),[Sinv](https://github.com/junzhezhang/shape-inversion) and [Pointnet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). We thank the authors for providing the source code.
