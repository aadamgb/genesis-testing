import argparse
import numpy as np
import torch
import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--time", type=float, default=1.0)
parser.add_argument("-c", "--cpu", action="store_true", default=False)
args = parser.parse_args()

DT = 1e-3
SPAN = 2.0                 # cloth edge length (mesh scale)
Z = 1.5                    # drone altitude
CLOTH_Z = Z - 0.4         # cloth just under the drones
RADIUS = 0.25              # corner patch radius

steps = int(args.time / DT)

# ---------------------------------------------------------------- helpers
def ppos(cloth):
    p = cloth.get_particles_pos()
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    return np.asarray(p).reshape(-1, 3)

def particles_near(cloth, target_xyz, radius):
    d = np.linalg.norm(ppos(cloth) - np.asarray(target_xyz), axis=1)
    return np.where(d < radius)[0].tolist()

# ---------------------------------------------------------------- init
gs.init(backend=gs.cpu if args.cpu else gs.cuda, precision="32")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=DT, substeps=1),
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

a300_1 = scene.add_entity(
    morph=gs.morphs.Drone(
        file="misc/urdf/a300.urdf",
        pos=(-(SPAN - 0.8) / 2, 0.0, Z),
        propellers_spin=(-1, -1, 1, 1),
    ),
    material=rigid_mat,
)
a300_2 = scene.add_entity(
    morph=gs.morphs.Drone(
        file="misc/urdf/a300.urdf",
        pos=((SPAN - 0.8) / 2, 0.0, Z),
        propellers_spin=(-1, -1, 1, 1),
    ),
    material=rigid_mat,
)

intruder = scene.add_entity(
    morph=gs.morphs.Drone(
        file="urdf/drones/cf2x.urdf",
        pos=(0.0, 0.5, Z - 0.5),
    ),
    material=rigid_mat,
)


cloth = scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(
        file="meshes/cloth.obj",
        scale=SPAN,
        pos=(0.0, 0.0, CLOTH_Z),
        euler=(90.0, 0.0, 0.0)
    ),
    surface=gs.surfaces.Default(color=(0.2, 0.4, 0.8, 1.0)),
)

scene.build(n_envs=0)

# ---------------------------------------------------------------- attach
p = ppos(cloth)
lo, hi = p.min(0), p.max(0)

left = (
    particles_near(cloth, (lo[0], lo[1], CLOTH_Z), RADIUS)
    + particles_near(cloth, (lo[0], hi[1], CLOTH_Z), RADIUS)
)
right = (
    particles_near(cloth, (hi[0], lo[1], CLOTH_Z), RADIUS)
    + particles_near(cloth, (hi[0], hi[1], CLOTH_Z), RADIUS)
)
assert left and right and not (set(left) & set(right)), (len(left), len(right))
print(f"left: {len(left)}  right: {len(right)}")

cloth.fix_particles_to_link(a300_1.link_start, particles_idx_local=left)
cloth.fix_particles_to_link(a300_2.link_start, particles_idx_local=right)

# ---------------------------------------------------------------- run
a300_1.set_quat([1.0, 0.0, 0.0, 0.0], zero_velocity=True)
a300_2.set_quat([1.0, 0.0, 0.0, 0.0], zero_velocity=True)
HOVER_RPM = 15502.5
actions = torch.tensor([0.0, 0.0, 0.0, 0.0], device=gs.device, dtype=gs.tc_float)
intruder_act = torch.tensor([0.0, 0.0, 0.03, 0.03], device=gs.device, dtype=gs.tc_float)
for i in range(steps):
    a300_1.set_propellers_rpm((1 + actions * 0.8) * HOVER_RPM)
    a300_2.set_propellers_rpm((1 + actions * 0.8) * HOVER_RPM)
    intruder.set_propellers_rpm((1 + intruder_act * 0.8) * HOVER_RPM)
    scene.step()