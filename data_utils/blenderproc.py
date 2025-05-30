import blenderproc as bproc
from pathlib import Path
import numpy as np
import argparse
import os, sys
import imageio
from PIL import Image
import bpy
from mathutils import Vector, Euler, Matrix
import png
sys.path.append(Path(__file__).parent.as_posix())

def render_blender_proc(cad_path, output_dir, obj_poses, img_size, intrinsic):
    bproc.init()

    cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(np.eye(4), ["X", "-Y", "-Z"])
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_intrinsics_from_K_matrix(
        intrinsic, img_size[1], img_size[0]
    )
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([1, -1, 1])
    light.set_energy(200)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([-1, -1, -1])
    light.set_energy(200)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([-1, 0, -1])
    light.set_energy(20)
    light.set_type("POINT")
    light.set_location([1, 0, 1])
    light.set_energy(20)

    objs = bproc.loader.load_obj(cad_path)
    obj_meshes = [obj for obj in objs if isinstance(obj.blender_obj.data, (bpy.types.Mesh))]
    obj_roots = [obj for obj in objs if obj.blender_obj.parent is None]
    
    bounds = np.concatenate([obj.get_bound_box() for obj in obj_meshes])  # N x 3
    bounds = np.min(bounds, axis=0), np.max(bounds, axis=0)
    scale = 1. / max(bounds[1] - bounds[0])
    center = (bounds[0] + bounds[1]) / 2. * scale
    
    parent_empty = bpy.data.objects.new("ParentEmpty", None)
    bpy.context.scene.collection.objects.link(parent_empty)

    # parent all root objects to the empty object
    for obj in obj_roots:
        obj.blender_obj.parent = parent_empty

    parent_empty.scale = parent_empty.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    parent_empty.matrix_world.translation[0] -= center[0]
    parent_empty.matrix_world.translation[1] -= center[1]
    parent_empty.matrix_world.translation[2] -= center[2]

    bproc.renderer.enable_distance_output(False)
    bproc.renderer.set_max_amount_of_samples(128)
    black_img = Image.new('RGB', (img_size[1], img_size[0]))
    
    mat = parent_empty.matrix_world.copy()
    for idx_frame, obj_pose in enumerate(obj_poses):
        parent_empty.matrix_world = Matrix(obj_pose) @ Matrix(mat)
        data = bproc.renderer.render()
        depth = bproc.postprocessing.dist2depth(data["distance"])[0]
        mask = np.uint8((depth < 1000) * 255)
        mask = Image.fromarray(mask)
        mask.save(os.path.join(output_dir, "mask_{:06d}.png".format(idx_frame)))

        rgb = Image.fromarray(np.uint8(data["colors"][0]))
        img = Image.composite(rgb, black_img, mask)
        img.save(os.path.join(output_dir, "color_{:06d}.png".format(idx_frame)))
        
        depth[depth >= 1000] = 0
        depth_mm = 1000. * depth
        depth_mm[depth_mm > 65535] = 65535
        im_uint16 = np.round(depth_mm).astype(np.uint16)

        # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
        w_depth = png.Writer(depth_mm.shape[1], depth_mm.shape[0], greyscale=True, bitdepth=16)
        with open(os.path.join(output_dir, "depth_{:06d}.png".format(idx_frame)), 'wb') as f:
            w_depth.write(f, np.reshape(im_uint16, (-1, depth_mm.shape[1])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cad_path', nargs='?', help="Path to the model file")
    # parser.add_argument('obj_pose', nargs='?', help="Path to the model file")
    parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved")
    parser.add_argument('disable_output', nargs='?', help="Disable output of blender")
    args = parser.parse_args()

    poses = np.load('data/obj_poses.npy')
    intrinsic = np.array([[35 * 512 / 32., 0., 256],  # to assign with objaverse
                            [0., 35 * 512 / 32., 256],
                            [0., 0., 1.]])
    img_size = [512, 512]
    is_tless = False

    if args.disable_output == "true":
        # redirect output to log file
        logfile = os.path.join(args.output_dir, 'render.log')
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)
    # scale_meter do not change the binary mask but recenter_origin change it
    render_blender_proc(args.cad_path, args.output_dir, poses[:], intrinsic=intrinsic, img_size=img_size)
    if args.disable_output == "true":
        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)
        os.system("rm {}".format(logfile))
