# DEDO  - Dynamic Environments with Deformable Objects
Dedo is a lightweight, deterministic, and customizable deformable object GyM environment aimed towards researchers in the reinforcement learning
and robotics community. 
The environment a set of every day tasks involving deformable objects such as hanging cloth, dressing a person, and buttoning buttons. 
We have provided examples for integrating two popular 
reinforcement learning libraries (stable_baselin3 and rllib) and an example for collecting and training a Variational Autoencoder with our environment. Dedo is easy to set up with very few dependencies,
highly parallelizable and supports a wide range of customizations: loading custom objects, adjusting texture and material properties. 

**Table of Contents:**<br />
[Installation](#install)<br />
[GettingStarted](#examples)<br />
[RL Examples](#rl)<br />
[Tasks](#tasks)<br />
[Customization](#custom)<br />



<a name="install"></a>
## Installation

Optional initial step: create a new conda environment with
`conda create --name dedo python=3.7` and activate it with
`conda activate dedo`. 
Conda is not strictly needed, alternatives like virtualenv can be used;
a direct install without using virtual environments is ok as well.

Python 3.8 or 3.7 should work, though on some cluster/remote machines we saw
that pybullet installs successfully with python3.7, but has trouble with 3.8.


```
git clone https://github.com/contactrika/dedo
cd dedo
pip install numpy  # important: Nessasary for compiling numpy-enabled PyBullet
pip install -e .
```

To enable recording/logging videos install ffmpeg:
```
sudo apt-get install ffmpeg
```

<a name="examples"></a>
## Getting started
To get started, one can run one of the following commands to visualize our environment

Hanging a bag
```
python -m dedo.demo --env=HangBag-v1 --viz --debug
```
Hanging an apron
```
python -m dedo.demo --env=HangGarment-v1 --cam_resolution 400 --viz --debug
```
* `dedo.demo` is the demo module
* `--env=HangGarment-v1` specifies the environment
* `--viz` enables the GUI
* `---debug` outputs additional information in the console
* `--cam_resolution 400` specifies the size of the output window


![misc/imgs/hang_task_ui.jpg](misc/imgs/hang_task_ui.jpg)


<a name="rl"></a>
## RL Examples

`dedo/run_rl_sb3.py` gives an example of how to train an RL
algorithm from Stable Baselines 3:

```
python -m dedo.run_rl_sb3 --env=HangGarment-v0 \
    --logdir=/tmp/dedo --num_play_runs=3 --viz --debug
```

`dedo/run_rllib.py` gives an example of how to train an RL
algorithm using RLLib:

```
python -m dedo.run_rllib --env=HangGarment-v0 \
    --logdir=/tmp/dedo --num_play_runs=3 --viz --debug
```

For documentation, please refer to [Argument References](../../wiki/Basic-commands) page in wiki

To launch the Tensorboard:
```
tensorboard --logdir=/tmp/dedo --bind_all --port 6006 \
  --samples_per_plugin images=1000
```

## SVAE Examples

`dedo/run_svae.py` gives an example of how to train an RL
algorithm from Stable Baselines 3:

```
python -m dedo.run_rl_sb3 --env=HangGarment-v0 \
    --logdir=/tmp/dedo --num_play_runs=3 --viz --debug
```

`dedo/run_rllib.py` gives an example of how to train an RL
algorithm from Stable Baselines 3:

```
python -m dedo.run_rl_sb3 --env=HangGarment-v0 \
    --logdir=/tmp/dedo --num_play_runs=3 --viz --debug
```

To launch the Tensorboard:
```
tensorboard --logdir=/tmp/dedo --bind_all --port 6006 \
  --samples_per_plugin images=1000
```


## Tasks:
We provide a set of 10 tasks involving deformable objects, most tasks contains 5 handmade deformable objects. 
There are also two procedurally generated tasks, `ButtonProc` and `HangProcCloth`, in which the deformable objects are procedurally generated. 
Furthermore, to improve generalzation, the `v0` of each task will randomizes textures and meshes.

All tasks have `-v1` and `-v2` with a particular choice of meshes and textures
that is not randomized. Most tasks have versions up to `-v5` with additional
mesh and texture variations.

Tasks with procedurally generated cloth (`ButtonProc` and `HangProcCloth`)
generate random cloth objects for all versions (but randomize textures only
in `v0`).

### HangBag
Over View Here
`HangBag-v0`: selects one of 108 bag meshes; randomized textures

`HangBag-v[1-3]`: three bag versions with textures shown below:

![misc/imgs/hang_bags_annotated.jpg](misc/imgs/hang_bags_annotated.jpg)


### HangGarment
`HangGarment-v0`: hang garment with randomized textures 
(a few examples below):

![misc/imgs/hang_garments_0.jpg](misc/imgs/hang_garments_0.jpg)

`HangGarment-v[1-5]`: 5 apron meshes and texture combos shown below:

![misc/imgs/hang_garments_5.jpg](misc/imgs/hang_garments_5.jpg)


`HangGarment-v[6-10]`: 5 shirt meshes and texture combos shown below:

![misc/imgs/hang_shirts_5.jpg](misc/imgs/hang_shirts_5.jpg)


### HangProcCloth

`HangProcCloth-v0`: random textures, 
procedurally generated cloth with 1 and 2 holes.

`HangProcCloth-v[1-2]`: same, but with either 1 or 2 holes

![misc/imgs/hang_proc_cloth.jpg](misc/imgs/hang_proc_cloth.jpg)

### Buttoning

`ButtonProc-v0`: randomized textures and procedurally generated cloth with 
2 holes, randomized hole/button positions.

`ButtonProc-v[1-2]`: procedurally generated cloth, 1 or two holes.

![misc/imgs/button_proc.jpg](misc/imgs/button_proc.jpg)

`Button-v0`: randomized textures, but fixed cloth and button positions.

`Button-v1`:  fixed cloth and button positions with one texture 
(see image below):

![misc/imgs/button.jpg](misc/imgs/button.jpg)


#### Hoop and Lasso

`Hoop-v0`, `Lasso-v0`: randomized textures

`Hoop-v1`, `Lasso-v1`: pre-selected textures

![misc/imgs/hoop_and_lasso.jpg](misc/imgs/hoop_and_lasso.jpg)


### Dress

`DressBag-v0`, `DressBag-v[1-5]`: demo for `-v1` shown below

![misc/imgs/dress_bag.jpg](misc/imgs/dress_bag.jpg)

Visualizations of the 5 backpack mesh and texture variants for `DressBag-v[1-5]`:

![misc/imgs/backpack_meshes.jpg](misc/imgs/backpack_meshes.jpg)

`DressGarment-v0`, `DressGarment-v[1-5]`: demo for `-v1` shown below

![misc/imgs/dress_garment.jpg](misc/imgs/dress_garment.jpg)

### Mask

`Mask-v0`, `Mask-v[1-5]`: a few texture variants shown below:

![misc/imgs/mask_0.jpg](misc/imgs/mask_0.jpg)



<a name="custom"></a>
## Customization

To load custom object you would first have to fill an entry in `DEFORM_INFO` in 
`task_info.py`. The key should the the `.obj` file path relative to `data/`:

```
DEFORM_INFO = {
...
    # An example of info for a custom item.
    'bags/custom.obj': {
        'deform_init_pos': [0, 0.47, 0.47],
        'deform_init_ori': [np.pi/2, 0, 0],
        'deform_scale': 0.1,
        'deform_elastic_stiffness': 1.0,
        'deform_bending_stiffness': 1.0,
        'deform_true_loop_vertices': [
            [0, 1, 2, 3]  # placeholder, since we don't know the true loops
        ]
    },
```

Then you can use `--override_deform_obj` flag:

```
python -m dedo.demo --env=HangBag-v0 --cam_resolution 200 --viz --debug \
    --override_deform_obj bags/custom.obj
```


For items not in `DEFORM_DICT` you will need to specify sensible defaults,
for example:

```
python -m dedo.demo --env=HangGarment-v0 --viz --debug \
  --override_deform_obj=generated_cloth/generated_cloth.obj \
  --deform_init_pos 0.02 0.41 0.63 --deform_init_ori 0 0 1.5708
```

Example of scaling up the custom mesh objects:
```
python -m dedo.demo --env=HangGarment-v0 --viz --debug \
   --override_deform_obj=generated_cloth/generated_cloth.obj \
   --deform_init_pos 0.02 0.41 0.55 --deform_init_ori 0 0 1.5708 \
   --deform_scale 2.0 --anchor_init_pos -0.10 0.40 0.70 \
   --other_anchor_init_pos 0.10 0.40 0.70
```
