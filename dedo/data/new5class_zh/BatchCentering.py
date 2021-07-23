import pymeshlab
import os

SRC_OBJ_ROOT = "obj_tri" # obj_sq

OBJ_LIST = ['apron', 'backpack', 'mask', 'lasso2d', 'lasso3d', 'tshirt']

# centering the tri mesh
for objclass in OBJ_LIST:
    # 5 obj for each class
    for i in range(5):
        if os.path.exists("{}_norm/{}".format(SRC_OBJ_ROOT, objclass)) is False:
            print("Create folder : {}_norm/{}".format(SRC_OBJ_ROOT, objclass))
            os.makedirs("{}_norm/{}".format(SRC_OBJ_ROOT, objclass))

        src_path = "{}/{}/{}_{}.obj".format(SRC_OBJ_ROOT, objclass, objclass, i)
        dst_path = "{}_norm/{}/{}_{}.obj".format(SRC_OBJ_ROOT, objclass, objclass, i)
        # load the mesh
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(src_path)
        ms.load_filter_script('./centering.mlx')
        ms.apply_filter_script()
        ms.save_current_mesh(dst_path)

