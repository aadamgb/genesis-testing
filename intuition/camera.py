import argparse
import genesis as gs
import torch
import numpy as np
import matplotlib.pyplot as plt
from genesis.utils.misc import tensor_to_array

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--time", type=float, default=15)
parser.add_argument("--every", type=int, default=100, help="save a frame every N steps")
args = parser.parse_args()

dt = 0.001
steps = int(args.time / dt)

gs.init(backend=gs.cuda)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=dt),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    show_viewer=True,
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=0.7,
        show_link_frame=True,
    ),
)

plane = scene.add_entity(gs.morphs.Plane())
a300 = scene.add_entity(gs.morphs.Drone(file="misc/urdf/a300.urdf",
                                        propellers_spin=(-1, -1, 1, 1)))

for (x, y), color in zip([(3.0, 0.0), (0.0, 3.0), (-3.0, 0.0), (0.0, -3.0)],
                         [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]):
    scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 2.0), pos=(x, y, 1.0), fixed=True),
        surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=color)),
    )

# FPV camera: pos and lookat are in the LINK frame once entity_idx is set
def fpv_offset_T(tilt_deg=22.0, fwd=0.05, up=0.02):
    """Body -> camera. Genesis cameras are OpenGL: -z forward, +y up, +x right."""
    # camera axes as columns, expressed in body frame (body: +x fwd, +z up)
    R = np.array([[0.0,  0.0, -1.0],    # x_cam = -body_y (right)
                  [-1.0, 0.0,  0.0],    # y_cam = +body_z (up)
                  [0.0,  1.0,  0.0]])   # z_cam = -body_x (back)
    t = np.deg2rad(tilt_deg)            # positive = tilt up
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(t), -np.sin(t)],
                   [0, np.sin(t),  np.cos(t)]])
    T = np.eye(4)
    T[:3, :3] = R @ Rx
    T[:3, 3] = (fwd, 0.0, up)
    return T

# cam = scene.add_camera(res=(320, 240), fov=90, GUI=True)
cam = scene.add_camera(res=(64, 64), fov=90, GUI=True)

fpv = scene.add_sensor(
    gs.sensors.RasterizerCameraOptions(
        res=(320, 240),          # (width, height)
        pos=(0.05, 0.0, 0.02),   # 5 cm forward, 2 cm up from body origin
        lookat=(1.0, 0.0, 0.4),  # forward, tilted ~22° up (typical racing quad)
        up=(0.0, 0.0, 1.0),
        fov=90.0,
        entity_idx=a300.idx,
        link_idx_local=0,        # base link
    ),
)



scene.build()
cam.attach(a300.base_link, fpv_offset_T())

a300.set_pos([0.0, 0.0, 0.5])
a300.set_quat([1.0, 0.0, 0.0, 0.0])

actions = torch.tensor([-1, -1, 0.7, 0.7], device=gs.device, dtype=gs.tc_float)

for i in range(steps):
    a300.set_propellers_rpm((1 + actions * 0.8) * 15502.5)
    scene.step()
    cam.move_to_attach()
    cam.render(rgb=True, depth=True)
    # if i % args.every == 0:
    #     rgb = fpv.read().rgb          # (H, W, 3) uint8, no env axis (unbatched build)
    #     plt.imsave(f"fpv_{i:06d}.png", tensor_to_array(rgb))
    
        