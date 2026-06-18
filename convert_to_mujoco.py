#!/usr/bin/env python3
"""Generate a runtime-friendly MuJoCo XML from `h2_model/urdf/h2.urdf`.

This keeps the existing project conventions used by `model/urdf/h2.xml`:
- 29 actuators in H2 joint order
- IMU / frame sensors with the expected names
- parent/child collision exclusions
- foot contact helper geoms

It also works around the currently missing `head_yaw_link.STL` export by using the
legacy mesh as a fallback during conversion.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


WRIST_JOINTS = {
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
}

FALLBACK_MESHES = {
    "head_yaw_link.STL": "model/meshes/head_yaw_Link.STL",
}

FOOT_CONTACT_SPECS = {
    "left_ankle_roll_link": {
        "name": "left_foot_contact",
        "type": "box",
        "size": "0.115 0.048 0.004",
        "pos": "0.037 0 -0.060",
        "contype": "0",
        "conaffinity": "1",
        "condim": "3",
        "friction": "2.0 0.5 0.1",
        "solref": "0.005 1",
        "solimp": "0.99 0.999 1e-05",
        "group": "1",
    },
    "right_ankle_roll_link": {
        "name": "right_foot_contact",
        "type": "box",
        "size": "0.115 0.048 0.004",
        "pos": "0.037 0 -0.060",
        "contype": "0",
        "conaffinity": "1",
        "condim": "3",
        "friction": "2.0 0.5 0.1",
        "solref": "0.005 1",
        "solimp": "0.99 0.999 1e-05",
        "group": "1",
    },
}

MATERIALS = {
    "grey_plastic": "0.79216 0.81961 0.93333 1",
    "default_material": "0.7 0.7 0.7 1",
    "collision_material": "1.0 0.28 0.1 0.9",
}


def parse_urdf(urdf_path: Path):
    tree = ET.parse(urdf_path)
    robot = tree.getroot()

    link_names = [link.get("name", "") for link in robot.findall("link") if link.get("name")]
    child_links = {
        child.get("link", "")
        for child in robot.findall("joint/child")
        if child.get("link")
    }
    root_link_name = next(
        (name for name in link_names if name not in child_links),
        link_names[0] if link_names else "base_link",
    )

    referenced_meshes = []
    for mesh in robot.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            referenced_meshes.append(Path(filename).name)

    joints = []
    parent_child_pairs = []
    for joint in robot.findall("joint"):
        joint_type = joint.get("type", "")
        if joint_type not in {"revolute", "continuous", "prismatic"}:
            continue

        name = joint.get("name")
        if not name:
            continue

        parent = joint.find("parent")
        child = joint.find("child")
        limit = joint.find("limit")
        effort = float(limit.get("effort", "0")) if limit is not None else 0.0

        joints.append({"name": name, "effort": effort})
        if parent is not None and child is not None:
            parent_child_pairs.append((parent.get("link", ""), child.get("link", "")))

    return robot.get("name", urdf_path.stem), root_link_name, referenced_meshes, joints, parent_child_pairs


def ensure_staged_meshes(stage_dir: Path, repo_root: Path, mesh_names: list[str], meshes_dir: Path):
    for mesh_path in meshes_dir.glob("*.STL"):
        target = stage_dir / mesh_path.name
        if not target.exists():
            target.symlink_to(mesh_path.resolve())

    missing = []
    for mesh_name in sorted(set(mesh_names)):
        staged = stage_dir / mesh_name
        if staged.exists():
            continue

        fallback_rel = FALLBACK_MESHES.get(mesh_name)
        if fallback_rel:
            fallback = repo_root / fallback_rel
            if fallback.exists():
                staged.symlink_to(fallback.resolve())
                continue

        missing.append(mesh_name)

    if missing:
        raise FileNotFoundError(
            "Missing meshes required for MuJoCo import: " + ", ".join(missing)
        )


def ensure_default_block(root: ET.Element):
    existing = root.find("default")
    if existing is not None:
        root.remove(existing)

    compiler = root.find("compiler")
    if compiler is None:
        root.insert(0, ET.Element("compiler", {"angle": "radian"}))
    else:
        compiler.set("angle", "radian")


def ensure_materials(root: ET.Element):
    asset = root.find("asset")
    if asset is None:
        insert_idx = 1
        compiler = root.find("compiler")
        if compiler is not None:
            insert_idx = list(root).index(compiler) + 1
        asset = ET.Element("asset")
        root.insert(insert_idx, asset)

    existing = {m.get("name") for m in asset.findall("material")}
    for name, rgba in MATERIALS.items():
        if name not in existing:
            ET.SubElement(asset, "material", {"name": name, "rgba": rgba})

    for mesh in asset.findall("mesh"):
        file_attr = mesh.get("file", "")
        mesh.set("file", f"../meshes/{Path(file_attr).name}")


def ensure_body_decorations(root: ET.Element, root_link_name: str):
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Generated MJCF has no <worldbody>.")

    root_body = worldbody.find(f"./body[@name='{root_link_name}']")
    if root_body is None:
        root_body = ET.Element(
            "body",
            {
                "name": root_link_name,
                "pos": "0.00000000 0.00000000 0.81159019",
                "quat": "1 0 0 0",
            },
        )
        ET.SubElement(root_body, "freejoint", {"name": "floating_base"})
        existing_children = list(worldbody)
        for child in existing_children:
            worldbody.remove(child)
            root_body.append(child)
        worldbody.append(root_body)
    elif root_body.find("freejoint") is None:
        root_body.insert(0, ET.Element("freejoint", {"name": "floating_base"}))

    for joint in root.findall(".//joint"):
        name = joint.get("name", "")
        if name == "floating_base":
            continue
        joint.set("ref", joint.get("ref", "0.0"))
        joint.set("damping", "0.05")
        joint.set("armature", "0.01")
        joint.set("frictionloss", "0.1" if name in WRIST_JOINTS else "0.2")

    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if name.endswith("_visual"):
            geom.set("material", "grey_plastic")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("group", "2")
        elif name.endswith("_collision"):
            geom.set("material", "collision_material")
            geom.set("condim", "3")
            geom.set("contype", "0")
            geom.set("conaffinity", "1")
            geom.set("priority", "1")
            geom.set("group", "1")
            geom.set("solref", "0.005 1")
            geom.set("solimp", "0.99 0.999 1e-05")
            geom.set("friction", "1 0.01 0.01")
        elif geom.get("type") == "mesh" and not name:
            # Nameless mesh geoms from URDF export: make visual-only.
            # Ground contact is handled exclusively by the foot contact box geoms.
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    for body_name, geom_attrs in FOOT_CONTACT_SPECS.items():
        body = root.find(f".//body[@name='{body_name}']")
        if body is not None and body.find(f"geom[@name='{geom_attrs['name']}']") is None:
            ET.SubElement(body, "geom", geom_attrs)

    extra_children = [
        ("site", {"name": "base_link_site", "pos": "0 0 0", "quat": "1 0 0 0"}),
        ("site", {"name": "imu", "pos": "0 0 0", "quat": "1 0 0 0", "size": "0.01"}),
        (
            "camera",
            {
                "name": "front_camera",
                "mode": "track",
                "fovy": "90.0",
                "quat": "4.329780281177467e-17 4.329780281177466e-17 0.7071067811865475 0.7071067811865476",
                "pos": "0.0 2.0 0.5",
            },
        ),
        (
            "camera",
            {
                "name": "side_camera",
                "mode": "track",
                "fovy": "90.0",
                "quat": "-0.5 -0.4999999999999999 0.5 0.5000000000000001",
                "pos": "-2.0 0.0 0.5",
            },
        ),
    ]

    for tag, attrs in extra_children:
        name = attrs.get("name")
        if name and root_body.find(f"{tag}[@name='{name}']") is None:
            ET.SubElement(root_body, tag, attrs)


def rebuild_actuators(root: ET.Element, joint_order: list[str], effort_by_joint: dict[str, float]):
    existing = root.find("actuator")
    if existing is not None:
        root.remove(existing)

    actuator = ET.SubElement(root, "actuator")
    for joint_name in joint_order:
        effort = effort_by_joint.get(joint_name, 0.0)
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": f"{joint_name}_ctrl",
                "joint": joint_name,
                "ctrlrange": f"-{effort:g} {effort:g}",
            },
        )


def rebuild_contact_excludes(root: ET.Element, parent_child_pairs: list[tuple[str, str]]):
    existing = root.find("contact")
    if existing is not None:
        root.remove(existing)

    body_names = {body.get("name") for body in root.findall(".//body") if body.get("name")}
    contact = ET.SubElement(root, "contact")
    seen = set()
    for parent, child in parent_child_pairs:
        key = (parent, child)
        if (
            not parent
            or not child
            or key in seen
            or parent not in body_names
            or child not in body_names
        ):
            continue
        ET.SubElement(contact, "exclude", {"body1": parent, "body2": child})
        seen.add(key)


def _torque_sensor_name(joint_name: str) -> str:
    if joint_name.endswith("_joint"):
        return joint_name[: -len("_joint")] + "_torque"
    return f"{joint_name}_torque"


def rebuild_sensors(root: ET.Element, joint_order: list[str]):
    existing = root.find("sensor")
    if existing is not None:
        root.remove(existing)

    sensor = ET.SubElement(root, "sensor")

    sensor.append(ET.Comment(f" Motor position sensors: sensordata[0..{len(joint_order)-1}] "))
    for joint_name in joint_order:
        ET.SubElement(sensor, "jointpos", {"name": f"{joint_name}_pos", "joint": joint_name})

    start = len(joint_order)
    end = 2 * len(joint_order) - 1
    sensor.append(ET.Comment(f" Motor velocity sensors: sensordata[{start}..{end}] "))
    for joint_name in joint_order:
        ET.SubElement(sensor, "jointvel", {"name": f"{joint_name}_vel", "joint": joint_name})

    start = 2 * len(joint_order)
    end = 3 * len(joint_order) - 1
    sensor.append(ET.Comment(f" Motor torque sensors: sensordata[{start}..{end}] "))
    for joint_name in joint_order:
        ET.SubElement(
            sensor,
            "jointactuatorfrc",
            {"name": _torque_sensor_name(joint_name), "joint": joint_name},
        )

    sensor.append(ET.Comment(" IMU sensors (named for bridge lookup) "))
    ET.SubElement(sensor, "framequat", {"name": "imu_quat", "objtype": "site", "objname": "imu"})
    ET.SubElement(sensor, "gyro", {"name": "imu_gyro", "site": "imu"})
    ET.SubElement(sensor, "accelerometer", {"name": "imu_acc", "site": "imu"})

    sensor.append(ET.Comment(" Frame position/velocity sensors "))
    ET.SubElement(sensor, "framepos", {"name": "frame_pos", "objtype": "site", "objname": "imu"})
    ET.SubElement(sensor, "framelinvel", {"name": "frame_vel", "objtype": "site", "objname": "imu"})


def convert(urdf_path: Path, output_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    meshes_dir = urdf_path.parent.parent / "meshes"
    model_name, root_link_name, referenced_meshes, urdf_joints, parent_child_pairs = parse_urdf(urdf_path)
    effort_by_joint = {entry["name"]: entry["effort"] for entry in urdf_joints}

    with tempfile.TemporaryDirectory(prefix="h2_mjcf_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        staged_urdf = tmpdir / urdf_path.name
        shutil.copy2(urdf_path, staged_urdf)
        ensure_staged_meshes(tmpdir, repo_root, referenced_meshes, meshes_dir)

        model = mujoco.MjModel.from_xml_path(str(staged_urdf))
        joint_order = [model.joint(i).name for i in range(model.njnt)]
        generated_xml = tmpdir / output_path.name
        mujoco.mj_saveLastXML(str(generated_xml), model)

        tree = ET.parse(generated_xml)
        root = tree.getroot()
        root.set("model", model_name)

        ensure_default_block(root)
        ensure_materials(root)
        ensure_body_decorations(root, root_link_name)
        rebuild_actuators(root, joint_order, effort_by_joint)
        rebuild_contact_excludes(root, parent_child_pairs)
        rebuild_sensors(root, joint_order)

        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=False)

    print(f"Generated {output_path}")
    print(f"  joints   : {len(joint_order)}")
    print(f"  actuators: {len(joint_order)}")
    print(f"  sensors  : {len(joint_order) * 3 + 5}")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=script_dir / "urdf" / "h2.urdf",
        help="Input URDF file (default: h2_model/urdf/h2.urdf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "urdf" / "h2.xml",
        help="Output MJCF file (default: h2_model/urdf/h2.xml)",
    )
    args = parser.parse_args()

    convert(args.urdf.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
