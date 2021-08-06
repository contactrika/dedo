# Dynamic Environments with Deformable Objects

TODO: port the rest of the code and add documentation

Workshop page with paper+poster: https://sites.google.com/nvidia.com/do-sim/posters


**Table of Contents:**<br />
[Installation](#install)<br />
[Basic Examples](#examples)<br />
[RL Examples](#rl)<br />

<a name="install"></a>
## Installation

Optional initial step: create a new conda environment with
`conda create --name dedo python=3.7` and activate it with
`conda activate dedo`. Conda is not strictly needed, _but python 3.7 is required_ as pybullet does not compile with numpy in python3.8.
alternatives like virtualenv can be used;
a direct install without using virtual environments is ok as well.

```
git clone https://github.com/contactrika/dedo
cd dedo
pip install numpy
pip install -e .
```

<a name="examples"></a>
### Basic Examples

```
python -m dedo.demo --env=HangBag-v0 --viz --debug
```

![misc/imgs/bag_begin.png](misc/imgs/bag_begin.png)
![misc/imgs/bag_end.png](misc/imgs/bag_end.png)

```
python -m dedo.demo --env=HangCloth-v0 --viz --debug
```

![misc/imgs/apron_begin.png](misc/imgs/apron_begin.png)
![misc/imgs/apron_end.png](misc/imgs/apron_end.png)


The above will only have anchor positions as the state (this is just for quick
testing). 

To get images as state use `--cam_resolution` flag as follows:

```
python -m dedo.demo --env=HangCloth-v0 --cam_resolution 200 --viz --debug
```

![misc/imgs/apron_rgb.png](misc/imgs/apron_rgb.png)

To load custom object you would first have to fill an entry in `DEFORM_INFO` in 
`task_info.py`. The key should the the `.obj` file path relative to `data/`:

```
DEFORM_INFO = {
...
    # An example of info for a custom item.
    'bags/bags_zehang/obj/normal/bag1-1.obj': {
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
    --override_deform_obj bags/bags_zehang/obj/normal/bag1-1.obj
```

![misc/imgs/bag_zehang.png](misc/imgs/bag_zehang.png)


For items not in `DEFORM_DICT` you will need to specify sensible defaults,
for example:

```
python -m dedo.demo --env=HangCloth-v0 --viz --debug \
  --override_deform_obj=generated_cloth/generated_cloth.obj \
  --deform_init_pos 0.02 0.41 0.63 --deform_init_ori 0 0 1.5708
```

Example of scaling up the custom mesh objects:
```
python -m dedo.demo --env=HangCloth-v0 --viz --debug \
   --override_deform_obj=generated_cloth/generated_cloth.obj \
   --deform_init_pos 0.02 0.41 0.55 --deform_init_ori 0 0 1.5708 \
   --deform_scale 2.0 --anchor_init_pos -0.10 0.40 0.70 \
   --other_anchor_init_pos 0.10 0.40 0.70
```

<a name="rl"></a>
## RL Examples

`dedo/rl_demo.py` gives an example of how to train an Rl
algorithm from Stable Baselines:

```
python -m dedo.rl_demo --env=HangCloth-v0 \
    --logdir=/tmp/dedo --num_play_runs=3 --viz --debug

tensorboard --logdir=/tmp/dedo --bind_all --port 6006 \
  --samples_per_plugin images=1000
```

After 5-10 minutes of training (on a laptop) on a simplified
env version where the anchor positions are given as
observations we get:

![misc/imgs/apron_ppo_play.gif](misc/imgs/apron_ppo_play.gif)

### More RL examples

### Buttoning
Envs that are ready: `ButtonSimple-v0` 
```
python -m dedo.rl_demo --env=ButtonSimple-v0 --logdir=/tmp/dedo --num_play_runs=1 --max_episode_len=800
```

### HangBag
Envs that are ready: `HangBag-v0`, `HangBag-v1`,`HangBag-v2`,`HangBag-v3`,`HangBag-v4`
```
python -m dedo.rl_demo --env=HangBag-v0  --logdir=/tmp/dedo --num_play_runs=1 ---max_episode_len=1500
```

### HangCloth
Envs that are ready: `HangCloth-v0`, `HangCloth-v1`,`HangCloth-v2`,`HangCloth-v3`,`HangCloth-v4`
```
python -m dedo.rl_demo --env=HangCloth-v5  --logdir=/tmp/dedo --num_play_runs=1 ---max_episode_len=1500
```

### Lasso
Envs that are ready: `Lasso-v0`
```
python -m dedo.rl_demo --env=Lasso-v0  --logdir=/tmp/dedo --num_play_runs=1 ---max_episode_len=1500
```