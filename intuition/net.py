import genesis as gs

gs.init()
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=2e-3,
        substeps=10,
    ),
    pbd_options=gs.options.PBDOptions(
        particle_size=1e-2,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.5, 0.0, 1.0),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
    ),
    vis_options=gs.options.VisOptions(
        rendered_envs_idx=[0],
    ),
    show_viewer=True,
)

frictionless_rigid = gs.materials.Rigid(
    needs_coup=True,
    coup_friction=0.0,
)

plane = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=frictionless_rigid,
)

obj = scene.add_entity(
    morph=gs.morphs.Sphere(
        radius=0.2,
        pos=(0.0, 0.0, 0.0),
        fixed=True,
    ),
    material=frictionless_rigid,
    surface=gs.surfaces.Default(
        color=(0.8, 0.2, 0.2, 1.0),
    ),
)

cloth = scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(
        file="misc/net.obj",
        scale=0.4,
        pos=(0.0, 0.0, 1.0),
        euler=(90.0, 0.0, 0.0),
    ),
    surface=gs.surfaces.Default(
        color=(0.2, 0.8, 0.2, 1.0),
    ),
)

scene.build(n_envs=0)

for i in range(1000):
    scene.step()