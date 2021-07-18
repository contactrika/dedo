# Dynamic Environments with Deformable Objects

TODO: port the rest of the code and add documentation

Workshop page with paper+poster: https://sites.google.com/nvidia.com/do-sim/posters


**Table of Contents:**<br />
[Installation](#install)<br />
[Examples](#examples)<br />

<a name="install"></a>
## Installation

Optional initial step: create a new conda environment with
`conda create --name dedo python=3.8` and activate it with
`conda activate dedo`. Conda is not strictly needed;
alternatives like virtualenv can be used;
a direct install without using virtual environments is ok as well.

```
git clone https://github.com/contactrika/dedo
cd dedo
pip install numpy
pip install -e .
```


### Examples

Env names are in the form: ```[task-name][state-type][Viz/Debug]-v[deformable-item-id]```,
e.g. ```DressTopo-v7``` for apron+figure scene.

```
python -m gym_bullet_deform.deform_env_demo --env_name=BasicPtsViz-v4
```

Use ```BasicPts``` for point cloud state, 
```BasicTopo``` for state with only topological features,
```BasicCombo``` for combined state, ```BasicFast``` to turn off topological
feature extraction (to simulate faster, e.g. for debugging).

Version specifies which cloth item to load from:
<br />
[gym-bullet-deform/gym_bullet_deform/utils/bags](gym-bullet-deform/gym_bullet_deform/data/bags)
<br />
[gym-bullet-deform/gym_bullet_deform/utils/cloth](gym-bullet-deform/gym_bullet_deform/data/cloth)
<br />
[gym-bullet-deform/gym_bullet_deform/utils/ropes](gym-bullet-deform/gym_bullet_deform/data/ropes)

Appending ```Viz``` to env name launches pybullet simulator GUI.

You can customize the initial cloth scale, position, orientation 
(in Euler angles) and change initial anchor position.

![misc/img/deform_env_demo/BasicTopoComboViz-v0](misc/img/deform_env_demo/BasicTopoViz-v0.png)

```
python -m gym_bullet_deform.deform_env_demo --env_name=BasicTopoViz-v0 \
--cloth_init_pos 0 0 0.22 --sim_frequency=240 \
--anchor_init_pos -0.05 0.035 0.45 \
--other_anchor_init_pos 0.05 0.035 0.45 \
--cam_outdir=/tmp/tmp_tmp
```

Some bags need more stiffness,
e.g. backpack (v1), laundry bag (v2), paperbag (v3):

![misc/img/deform_env_demo/BasicTopoComboViz-v1](misc/img/deform_env_demo/BasicTopoViz-v1.png)
![misc/img/deform_env_demo/BasicTopoComboViz-v2](misc/img/deform_env_demo/BasicTopoViz-v2.png)
![misc/img/deform_env_demo/BasicTopoComboViz-v3](misc/img/deform_env_demo/BasicTopoViz-v3.png)

```
python -m gym_bullet_deform.deform_env_demo \
--env_name=BasicTopoViz-v1 --cloth_init_pos 0 0 0.3 \
--anchor_init_pos 0.1 -0.04 0.45 --other_anchor_init_pos 0.1 0.04 0.45 \
--cloth_elastic_stiffness=50 --cloth_bending_stiffness=50 \
--sim_frequency=500
```

```
python -m gym_bullet_deform.deform_env_demo \
--env_name=BasicTopoViz-v2 --cloth_init_pos 0 0 0.3 \
--anchor_init_pos 0.1 -0.04 0.45 --other_anchor_init_pos 0.1 0.04 0.45 \
--cloth_elastic_stiffness=50 --cloth_bending_stiffness=50 \
--sim_frequency=500
```

```
python -m gym_bullet_deform.deform_env_demo \
--env_name=BasicFastViz-v3 --cloth_init_pos 0 0 0.37 \
--anchor_init_pos 0.05 -0.04 0.45 --other_anchor_init_pos 0.05 0.04 0.45 \
--cloth_elastic_stiffness=50 --cloth_bending_stiffness=50 \
--sim_frequency=500
```


To record point clouds to separate analysis turn on ```--cam_outdir```
See more arguments in [gym-bullet-deform/gym_bullet_deform/utils/args.py](gym-bullet-deform/gym_bullet_deform/utils/args.py)

Ultimately the following scenes/tasks will be available: 
```Basic```, ```Dress```, ```Hang```,
```Hook```, ```Hoops```, ```Lasso```


### More Examples

Red lines that form the triangle (with inner lines meeting in the middle)
show the vertices of the triangle of the loop with the longest lifetime.

Green dot is the target: when the center of the red triangle is aligned
with the green dot we will have the reward=0 (o/w reward is the distance
of the center of the triangle to the green dot target).

You can select various cloth items by choosing different versions: i.e.
HangComboViz-v1 will select an apron, HangComboViz-v6 will select a laundry bag,
and so on. Filenames of the cloths and the corresponding version ids are
printed to STDOUT when you launch the sim.

#### Dress

Apron with one main loop approaching a figure:

![misc/img/deform_env_demo/DressComboViz-v6_0](misc/img/deform_env_demo/DressComboViz-v6_0.png)
![misc/img/deform_env_demo/DressComboViz-v6_1](misc/img/deform_env_demo/DressComboViz-v6_1.png)

```
python -m gym_bullet_deform.deform_env_demo --env_name=DressComboViz-v6 \
--num_runs=10 --sim_frequency=240 --cloth_scale=1.0 \
--cloth_init_pos 0.0 0.3 0.5 --cloth_init_ori 0 0 1.57 \
--anchor_init_pos -0.04 0.27 0.72 --other_anchor_init_pos 0.04 0.27 0.72
```

Apron with one main loop falling onto a hanger:

![misc/img/deform_env_demo/HangPtsViz-v6](misc/img/deform_env_demo/HangPtsViz-v6.png)
![misc/img/deform_env_demo/HangPtsViz-v6_other](misc/img/deform_env_demo/HangPtsViz-v6_other.png)

```
python -m gym_bullet_deform.deform_env_demo --env_name=HangComboViz-v6 \
--num_runs=10 --sim_frequency=240 --cloth_scale=0.8 \
--cloth_init_pos 0.0 0.03 0.45 --cloth_init_ori 0 0 1.57 \
--anchor_init_pos -0.04 0.03 0.62 --other_anchor_init_pos 0.04 0.03 0.62
```

#### Hang

Wide shirt falling onto a hanger:

![misc/img/deform_env_demo/HangComboViz-v10_0](misc/img/deform_env_demo/HangComboViz-v10_0.png)
![misc/img/deform_env_demo/HangComboViz-v10_1](misc/img/deform_env_demo/HangComboViz-v10_1.png)
![misc/img/deform_env_demo/HangComboViz-v10_2](misc/img/deform_env_demo/HangComboViz-v10_2.png)
![misc/img/deform_env_demo/HangComboViz-v10_3](misc/img/deform_env_demo/HangComboViz-v10_3.png)

```
python -m gym_bullet_deform.deform_env_demo --env_name=HangComboViz-v10 \
--num_runs=10 --sim_frequency=240 --cloth_scale=0.8 \
--cloth_init_pos 0.0 0.03 0.45 --cloth_init_ori 0 0 1.57 \
--anchor_init_pos -0.04 0.03 0.57 --other_anchor_init_pos 0.04 0.03 0.57
```

#### Hook

Apron with two loops falling onto a hook:

![misc/img/deform_env_demo/HookComboViz-v7_0](misc/img/deform_env_demo/HookComboViz-v7_0.png)
![misc/img/deform_env_demo/HookComboViz-v7_1](misc/img/deform_env_demo/HookComboViz-v7_1.png)


```
python -m gym_bullet_deform.deform_env_demo --env_name=HookComboViz-v7 \
--num_runs=10 --sim_frequency=240 --cloth_scale=0.8 \
--cloth_init_pos 0.0 0.27 0.45 --cloth_init_ori 0 0 1.57 \
--anchor_init_pos -0.04 0.25 0.62 --other_anchor_init_pos 0.04 0.25 0.62
```

Bag falling onto a hook:

![misc/img/deform_env_demo/HookComboViz-v5_0](misc/img/deform_env_demo/HookComboViz-v5_0.png)
![misc/img/deform_env_demo/HookComboViz-v5_1](misc/img/deform_env_demo/HookComboViz-v5_1.png)
![misc/img/deform_env_demo/HookComboViz-v5_2](misc/img/deform_env_demo/HookComboViz-v5_2.png)
![misc/img/deform_env_demo/HookComboViz-v5_3](misc/img/deform_env_demo/HookComboViz-v5_3.png)
![misc/img/deform_env_demo/HookComboViz-v5_4](misc/img/deform_env_demo/HookComboViz-v5_4.png)

```
python -m gym_bullet_deform.deform_env_demo --env_name=HookComboViz-v5 \
--num_runs=10 --sim_frequency=240 --cloth_scale=1.2 \
--cloth_init_pos 0.0 0.32 0.45 --cloth_init_ori 0 -0.52 1.57 \
--anchor_init_pos -0.04 0.22 0.58 --other_anchor_init_pos 0.04 0.22 0.58
```

#### Hoop

Putting a hoop onto a pole:

![misc/img/deform_env_demo/HoopComboViz-v0_far](misc/img/deform_env_demo/HoopComboViz-v0_far.png)


```
python -m gym_bullet_deform.deform_env_demo --env_name=HoopComboViz-v0 \
--num_runs=10 --sim_frequency=240 --cloth_scale=1.0 \
--cloth_init_pos 0.2 0.3 0.5 --cloth_init_ori 0 1.57 0 \
--anchor_init_pos 0.15 0.3 0.5 --other_anchor_init_pos 0.25 0.3 0.5
```

Hoop starting over a pole:

![misc/img/deform_env_demo/HoopComboViz-v0_0](misc/img/deform_env_demo/HoopComboViz-v0_0.png)
![misc/img/deform_env_demo/HoopComboViz-v0_1](misc/img/deform_env_demo/HoopComboViz-v0_1.png)
![misc/img/deform_env_demo/HoopComboViz-v0_2](misc/img/deform_env_demo/HoopComboViz-v0_2.png)


```
python -m gym_bullet_deform.deform_env_demo --env_name=HoopComboViz-v0 \
--num_runs=10 --sim_frequency=240 --cloth_scale=1.0 \
--cloth_init_pos 0.0 0.0 0.5 --cloth_init_ori 0 1.57 0 \
--anchor_init_pos -0.05 0.0 0.5 --other_anchor_init_pos 0.05 0.0 0.5
```

Use ```HoopComboViz-v1``` for a thinner hoop.

![misc/img/deform_env_demo/HoopComboViz-v1_0](misc/img/deform_env_demo/HoopComboViz-v1_0.png)
![misc/img/deform_env_demo/HoopComboViz-v1_1](misc/img/deform_env_demo/HoopComboViz-v1_1.png)
![misc/img/deform_env_demo/HoopComboViz-v1_3](misc/img/deform_env_demo/HoopComboViz-v1_3.png)


Use ```HoopComboViz-v2``` for a rope hoop.

![misc/img/deform_env_demo/HoopComboViz-v2_0](misc/img/deform_env_demo/HoopComboViz-v2_0.png)
![misc/img/deform_env_demo/HoopComboViz-v2_1](misc/img/deform_env_demo/HoopComboViz-v2_1.png)
![misc/img/deform_env_demo/HoopComboViz-v2_2](misc/img/deform_env_demo/HoopComboViz-v2_2.png)



#### Lasso

Lasso from a rope object:

![misc/img/deform_env_demo/LassoComboViz-v0_0](misc/img/deform_env_demo/LassoComboViz-v0_0.png)
![misc/img/deform_env_demo/LassoComboViz-v0_1](misc/img/deform_env_demo/LassoComboViz-v0_1.png)
![misc/img/deform_env_demo/LassoComboViz-v0_2](misc/img/deform_env_demo/LassoComboViz-v0_2.png)


```
python -m gym_bullet_deform.deform_env_demo --env_name=LassoComboViz-v0 \
--num_runs=10 --sim_frequency=240 --cloth_scale=1.0 \
--cloth_init_pos 0.0 0.0 0.5 --cloth_init_ori 1.57 0 0 \
--anchor_init_pos 0.06 -0.15 0.5 --other_anchor_init_pos 0.1 0.0 0.5
```

Lasso-like object (mesh from a rigid object loaded as SoftBody):

![misc/img/deform_env_demo/LassoComboViz-v1_0](misc/img/deform_env_demo/LassoComboViz-v1_0.png)
![misc/img/deform_env_demo/LassoComboViz-v1_1](misc/img/deform_env_demo/LassoComboViz-v1_1.png)
![misc/img/deform_env_demo/LassoComboViz-v1_2](misc/img/deform_env_demo/LassoComboViz-v1_2.png)


```
python -m gym_bullet_deform.deform_env_demo --env_name=LassoComboViz-v1 \
--num_runs=10 --sim_frequency=240 --cloth_scale=1.0 \
--cloth_init_pos 0.0 0.0 0.5 --cloth_init_ori 1.57 0 0 \
--anchor_init_pos 0.0 -0.3 0.5 --other_anchor_init_pos 0.0 -0.2 0.5
```

Use ```--cloth_init_ori 0 1.57 0``` to rotate around the long axis of the main loop.

Use ```LassoComboViz-v2``` for lasso with a sparser rope mesh.


#### Button

Note: topo only works when we omit rigid object points
(with ```--cam_resolution=1```)

![misc/img/deform_env_demo/ButtonComboViz-v0_0](misc/img/deform_env_demo/ButtonComboViz-v0_0.png)
![misc/img/deform_env_demo/ButtonComboViz-v0_1](misc/img/deform_env_demo/ButtonComboViz-v0_1.png)
![misc/img/deform_env_demo/ButtonComboViz-v0_2](misc/img/deform_env_demo/ButtonComboViz-v0_2.png)

```
python -m gym_bullet_deform.deform_env_demo --env_name=ButtonComboViz-v0 \
--num_runs=10 --sim_frequency=240 --cloth_scale=1.28 \
--cloth_init_pos -0.05 0 0 --cloth_init_ori 0 0 0 \
--cam_outdir=/tmp/tmp_tmp --cam_resolution=1
```


### RL with (ray)RLLib

First make sure dependencies are installed:
```
pip install tensorflow-gpu
cd rl-top-euc
pip install -e .
```

A basic test for training with off-policy ```Apex``` algorithm
for the simplest task of learning to keep the anchors at 0.5m height
(just learn to counteract gravity and cloth object pull on the anchors).

Without the cloth, PPO and ApexDDPG learn reasonably well
(```--cloth_init_pos 0.0 0.0 0.1``` set s.t. the anchors do not grab to cloth,
so it just falls at the beginning of the episode and anchors just need to
stay up by themselves in the air).

```
python -m rl_top_euc.rl.rllib_aux --rl_algo=ApexDDPG --gpus='7' --ncpus=8 \
--num_envs_per_worker=1 --env_name=BasicFast-v25 --sim_frequency=240 \
--cloth_scale=1.0 --anchor_tgt_z=0.50 \
--cloth_init_pos 0.0 0.0 0.1 --cloth_init_ori 0 1.57 0 \
--anchor_init_pos -0.05 0.0 0.5 --other_anchor_init_pos 0.05 0.0 0.5
```

```--gpus``` takes a list of GPU IDs, e.g.: ```--gpus='6,7'```

```--rl_algo``` takes the name of the algo; so far ```PPO``` and ```ApexDDPG``` run ok.

RLLib will use TF, but can set ```--use_pytorch``` to use pytorch 
(though TF-based implementations are more stable).

Parameters relevant for RL training can be set here:
https://github.com/contactrika/rl-top-euc/blob/master/rl_top_euc/rl/rllib_aux.py#L103

RLLib documentation (not detailed enough, but does give an overview):
https://docs.ray.io/en/master/rllib-algorithms.html

Training should output something like this periodically:
Number of trials: 1 (1 RUNNING)
```
+-------------------------------------+----------+----------------------+--------+------------------+--------+----------+
| Trial name                          | status   | loc                  |   iter |   total time (s) |     ts |   reward |
|-------------------------------------+----------+----------------------+--------+------------------+--------+----------|
| APEX_DDPG_BasicFast-v25_c4953_00000 | RUNNING  | 130.237.218.51:15934 |     29 |          938.959 | 194800 |  48.6403 |
+-------------------------------------+----------+----------------------+--------+------------------+--------+----------+
```

Tensorboard training dashboard: ```tensorboard --logdir=~/ray_results --port=6007```
(access with ssh and port forwarding: ```ssh -L 6007:127.0.0.1:6007 user@server```)


Visualize the learned policy:

```
python -m rl_top_euc.rl.rllib_aux --rl_algo=PPO --gpus='7' --ncpus=8 \
--num_envs_per_worker=1 --env_name=BasicFast-v25 --sim_frequency=240 \
--cloth_scale=1.0 --anchor_tgt_z=0.50 
--cloth_init_pos 0.0 0.0 0.1 --cloth_init_ori 0 1.57 0 \
--anchor_init_pos -0.05 0.0 0.5 --other_anchor_init_pos 0.05 0.0 0.5 \
--play 10 --load_checkpt \
~/ray_results/APEX_DDPG/APEX_DDPG_BasicFast-v25_0_2020-08-05_06-51-59hp6tw0sp/checkpoint_30/checkpoint-30
```

With the cloth both learn very slowly, so either sim is not very stable,
or this is just a hard task. Maybe including cloth pointcloud would help.

![misc/img/deform_env_demo/BasicFast-v25](misc/img/deform_env_demo/BasicFast-v25.png)


```
python -m rl_top_euc.rl.rllib_aux --rl_algo=PPO --gpus='7' --ncpus=8 \
--num_envs_per_worker=1 --env_name=BasicFast-v25 --sim_frequency=240 \
--cloth_scale=1.0 --anchor_tgt_z=0.50 
--cloth_init_pos 0.0 0.0 0.5 --cloth_init_ori 0 1.57 0 \
--anchor_init_pos -0.05 0.0 0.5 --other_anchor_init_pos 0.05 0.0 0.5
```


