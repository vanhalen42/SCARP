# SCARP: 3D Shape Completion in ARbitrary Poses for Improved Grasping

[Bipasha Sen](https://bipashasen.github.io/)<sup>\*1</sup>,
[Aditya Agarwal](http://skymanaditya1.github.io/)<sup>\*1</sup>,
[Gaurav Singh](https://www.linkedin.com/in/gaurav-singh-448363207/)<sup>1*</sup>,
[Brojeshwar Bhowmick](https://scholar.google.co.in/citations?user=Eqf8NrEAAAAJ&hl=en)<sup>2</sup>,
[Srinath Sridhar](https://cs.brown.edu/people/ssrinath/)<sup>3</sup>,
[Madhava Krishna](https://www.iiit.ac.in/people/faculty/mkrishna/)<sup>1</sup><br>
<sup>1</sup>International Institute of Information Technology, Hyderabad, <sup>2</sup>TCS Research India, <sup>3</sup>Brown University

<sup>\*</sup>denotes equal contribution

This is the official implementation of the paper "*SCARP: 3D Shape Completion in ARbitrary Poses for Improved Grasping*" **accepted** at ICRA 2023.

**This work was featured as "*The Publication of the Week*" by Weekly Robotics [**here**](https://www.weeklyrobotics.com/weekly-robotics-231)**

<img src="./results/result1.gif">
<!-- <img src="./results/result2.gif"> -->

For more results, information, and details visit our [**project page**](https://bipashasen.github.io/scarp) and read our [**paper**](https://arxiv.org/abs/2301.07213).


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

## Real World Results
<img src='./results/real1.gif'>
<!-- <img src='./results/real2.gif'> -->

## Acknowledgement

Some parts of the code are insipired and borrowed from [ConDor](https://github.com/brown-ivl/ConDor),[Sinv](https://github.com/junzhezhang/shape-inversion) and [Pointnet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). We thank the authors for providing the source code.


## Citation
If you find our work useful in your research, please cite:
```
@INPROCEEDINGS{10160365,
  author={Sen, Bipasha and Agarwal, Aditya and Singh, Gaurav and B., Brojeshwar and Sridhar, Srinath and Krishna, Madhava},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={SCARP: 3D Shape Completion in ARbitrary Poses for Improved Grasping}, 
  year={2023},
  volume={},
  number={},
  pages={3838-3845},
  doi={10.1109/ICRA48891.2023.10160365}
}
```

## Contact
If you have any questions, please feel free to email the authors.

Bipasha Sen: bipasha.sen@research.iiit.ac.in <br>
Aditya Agarwal: aditya.ag@research.iiit.ac.in <br>
Gaurav Singh: gaurav.si@research.iiit.ac.in <br>
