import argparse
import numpy as np
import torch
import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--time", type=float, default=1.0)
parser.add_argument("-c", "--cpu", action="store_true", default=False)
args = parser.parse_args()

DT = 0.01
L = 1.0                 
Z = 1.5  
HOVER_RPM = 8120.65 * 1.05                 

steps = int(args.time / DT)


# ---------------------------------------------------------------- init
gs.init(backend=gs.cpu if args.cpu else gs.cuda, precision="32")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=DT, substeps=10),
    pbd_options=gs.options.PBDOptions(particle_size=1e-2),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.0, -2.0, 2.0),
        camera_lookat=(0.0, 0.0, 1.0),
        camera_fov=40,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(show_world_frame=True, world_frame_size=0.7),
    show_viewer=True,
)

# ---------------------------------------------------------------- entities
rigid_mat = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)

scene.add_entity(gs.morphs.Plane(), rigid_mat)

bambi_1 = scene.add_entity(
    morph=gs.morphs.Drone(
        file="misc/urdf/bros300.urdf",
        pos=(- L / 2, 0.0, Z),
        propellers_spin=(-1, -1, 1, 1),
    ),
    material=rigid_mat,
)
bambi_2 = scene.add_entity(
    morph=gs.morphs.Drone(
        file="misc/urdf/bros300.urdf",
        pos=(L / 2, 0.0, Z),
        propellers_spin=(-1, -1, 1, 1),
    ),
    material=rigid_mat,
)

rod = scene.add_entity(
    morph=gs.morphs.Cylinder(
        radius=0.01,
        height= L + 0.1,
        euler=(0, 90, 0),
        pos=(0, 0, Z - 0.16),
    ),
    material=rigid_mat,
    surface=gs.surfaces.Default(color=(0.2, 0.4, 0.8, 1.0)),
)

net = scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(
        file="misc/net.obj",
        scale=0.5,
        pos=(0.0, 0.0, 0.82),
        euler=(180.0, 0.0, 0.0),
    ),
    surface=gs.surfaces.Default(
        color=(0.2, 0.6, 0.2, 1.0),
    ),
)

intruder = scene.add_entity(
    morph=gs.morphs.Drone(
        file="urdf/drones/cf2x.urdf",
        pos=(0.0, 0.5, Z - 0.5),
    ),
    material=rigid_mat,
)



scene.build(n_envs=0)

solver  = scene.sim.rigid_solver
tip1    = bambi_1.get_link("segment_10_cylinder")
tip2    = bambi_2.get_link("segment_10_cylinder")
rod_lnk = rod.base_link

solver.add_weld_constraint(tip1.idx, rod_lnk.idx)
solver.add_weld_constraint(tip2.idx, rod_lnk.idx)

P = net.get_particles_pos()                                
top = torch.where(P[:, 2] > P[:, 2].max() - 0.02)[0]      
net.fix_particles_to_link(rod_lnk.idx, particles_idx_local=top.tolist())

# ---------------------------------------------------------------- 
bambi_1.set_quat([1.0, 0.0, 0.0, 0.0], zero_velocity=True)
bambi_2.set_quat([1.0, 0.0, 0.0, 0.0], zero_velocity=True)

actions = torch.tensor([0.02, 0.02, 0.0, 0.0], device=gs.device, dtype=gs.tc_float)
intruder_act = torch.tensor([0.0, 0.0, 0.03, 0.03], device=gs.device, dtype=gs.tc_float)
for i in range(steps):
    bambi_1.set_propellers_rpm((1 + actions * 0.8) * HOVER_RPM )
    # bambi_2.set_propellers_rpm((1 + actions * 0.8) * HOVER_RPM )
    bambi_2.set_propellers_rpm((1 + intruder_act * 0.8) * HOVER_RPM )
    intruder.set_propellers_rpm((1 + intruder_act * 0.8) * 15502.5)
    scene.step()