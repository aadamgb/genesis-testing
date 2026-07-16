import argparse
import genesis as gs
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--time", type=int, default=15)
args = parser.parse_args()

steps = int(args.time / 0.001)
gs.init(backend=gs.cuda)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.001),
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
racer = scene.add_entity(gs.morphs.Drone(file="urdf/drones/racer.urdf"))
a300 = scene.add_entity(gs.morphs.Drone(file="misc/urdf/a300.urdf",
                                        propellers_spin=(-1, -1, 1, 1)))
crazy_fly = scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

scene.build()

racer.set_pos([0.0, -0.5, 2.0], zero_velocity=True)
racer.set_quat([1.0, 0.0, 0.0, 0.0], zero_velocity=True)

a300.set_pos([0.0, 0, 2.0], zero_velocity=True)
a300.set_quat([1.0, 0.0, 0.0, 0.0], zero_velocity=True)

crazy_fly.set_pos([0.0, 0.5, 2.0], zero_velocity=True)
crazy_fly.set_quat([1.0, 0.0, 0.0, 0.0], zero_velocity=True)
# actions = torch.zeros((1, 4), device=gs.device, dtype=gs.tc_float)
actions = torch.tensor([1.0, 1.0, 0.0, 0.0], device=gs.device, dtype=gs.tc_float)
for _ in range(steps ):
    racer.set_propellers_rpm((1 + 0 * 0.8) * 15502.5)
    a300.set_propellers_rpm((1 + actions * 0.8) * 15502.5)
    # a300.set_propellers_rpm((1 + 0.0 * 0.8) * 15502.5)
    crazy_fly.set_propellers_rpm((1 + 0 * 0.8) * 14475.81)
    scene.step()