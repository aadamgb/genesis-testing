import genesis as gs

gs.init(backend=gs.cuda)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    show_viewer=True,
)

plane = scene.add_entity(gs.morphs.Plane())
racer = scene.add_entity(gs.morphs.Drone(file="urdf/drones/racer.urdf"))
a300 = scene.add_entity(gs.morphs.Drone(file="urdf/drones/a300.urdf"))
crazy_fly = scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

scene.build()

racer.set_pos([0.0, -0.5, 2.0], zero_velocity=True)
racer.set_quat([1.0, 0.0, 0.0, 0.0], zero_velocity=True)

a300.set_pos([0.0, 0, 2.0], zero_velocity=True)
a300.set_quat([1.0, 0.0, 0.0, 0.0], zero_velocity=True)

crazy_fly.set_pos([0.0, 0.5, 2.0], zero_velocity=True)
crazy_fly.set_quat([1.0, 0.0, 0.0, 0.0], zero_velocity=True)
for _ in range(1000):
    racer.set_propellers_rpm((1 + 0.0 * 0.8) * 15502.5)
    a300.set_propellers_rpm((1 + 0.0 * 0.8) * 15502.5)
    crazy_fly.set_propellers_rpm((1 + 0.0 * 0.8) * 14475.81)
    scene.step()