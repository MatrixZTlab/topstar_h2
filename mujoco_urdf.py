import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("urdf/h2.xml")

# Topology
print(f"Bodies: {model.nbody}, Joints: {model.njnt}, Geoms: {model.ngeom}")

for i in range(model.nbody):
    name = model.body(i).name
    parent_id = model.body(i).parentid
    parent_name = model.body(parent_id).name if parent_id >= 0 else "world"
    pos = model.body(i).pos
    print(f"  [{i}] {parent_name} → {name}  pos={np.round(pos, 4)}")

# Joint info
for i in range(model.njnt):
    j = model.joint(i)
    print(f"Joint {j.name}: type={j.type}, range={j.range}, axis={j.axis}")

# Inertia
for i in range(model.nbody):
    b = model.body(i)
    print(f"{b.name}: mass={b.mass[0]:.4f}, inertia={b.inertia}")