import bpy
import os
import time
from os.path import join

datapath = '/media/pyshi/newssd/sewing/'

# Delete everything in the scene
def delete_all():
    bpy.ops.object.mode_set(mode='OBJECT')
    for key, scene_obj in dict(bpy.data.objects).items():
        print('deleteing',key,scene_obj)

        bpy.data.objects.remove(scene_obj, do_unlink=True)

        # scene_obj.select_set(True)
        # bpy.context.view_layer.objects.active = scene_obj
        # bpy.ops.object.delete()

import_dir = join(datapath, 'clean/')
files = os.listdir(import_dir)
files_len = len(files)
start_time = time.time()
for i, file_name in enumerate(files):
    delete_all()

    # Sanity check
    if not file_name.endswith('.obj'): continue

    # Import
    in_file = join(import_dir, file_name)
    bpy.ops.import_scene.obj(filepath=in_file)

    # edit mode in the first object
    objs = dict(bpy.data.objects)
    obj = list(objs.values())[0]
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    v_len = len(obj.data.vertices)
    print('file', len(obj.data.vertices))


    if v_len < 5000:
        ratio = 0.1
    # Decimate!
    status = bpy.ops.mesh.decimate(ratio=0.05)

    # Export
    outfile = join(datapath, 'simplified', f'{v_len:05d}_' + file_name)
    bpy.ops.export_scene.obj(filepath=outfile)

    # Stats
    elapsed = time.time() - start_time
    remaining = (files_len - i+1) / (i+1) * elapsed
    print(f'file {i}/{files_len}, time elapsed: {int(elapsed)}, est-remaining: {int(remaining)}')