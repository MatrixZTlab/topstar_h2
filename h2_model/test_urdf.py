"""Extract and display URDF topology, joint parameters, and inertial properties.
No external dependencies — uses stdlib xml.etree.ElementTree only.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict

URDF_PATH = "urdf/h2.urdf"
# URDF_PATH = "../../unitree_ros/robots/g1_description/g1_29dof.urdf"
# URDF_PATH = "../h2_model/urdf/h2.urdf"

def parse_urdf(path):
    tree = ET.parse(path)
    root = tree.getroot()

    links = {}
    joints = {}

    # --- Links ---
    for link in root.findall("link"):
        name = link.get("name")
        info = {"name": name, "mass": None, "inertia": None, "com": None}

        inertial = link.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            origin_el = inertial.find("origin")
            inertia_el = inertial.find("inertia")

            if mass_el is not None:
                info["mass"] = float(mass_el.get("value"))
            if origin_el is not None:
                info["com"] = origin_el.get("xyz")
            if inertia_el is not None:
                info["inertia"] = {k: float(inertia_el.get(k, 0))
                                   for k in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]}
        links[name] = info

    # --- Joints ---
    for joint in root.findall("joint"):
        name = joint.get("name")
        jtype = joint.get("type")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")

        origin_el = joint.find("origin")
        axis_el = joint.find("axis")
        limit_el = joint.find("limit")

        info = {
            "name": name,
            "type": jtype,
            "parent": parent,
            "child": child,
            "origin_xyz": origin_el.get("xyz") if origin_el is not None else "0 0 0",
            "origin_rpy": origin_el.get("rpy") if origin_el is not None else "0 0 0",
            "axis": axis_el.get("xyz") if axis_el is not None else "0 0 1",
            "limit": None,
        }

        if limit_el is not None:
            info["limit"] = {
                "lower": float(limit_el.get("lower", 0)),
                "upper": float(limit_el.get("upper", 0)),
                "effort": float(limit_el.get("effort", 0)),
                "velocity": float(limit_el.get("velocity", 0)),
            }

        joints[name] = info

    return links, joints


def build_tree(joints):
    """Build parent→[children] map and find root link."""
    children = defaultdict(list)
    all_children = set()
    for j in joints.values():
        children[j["parent"]].append((j["child"], j["name"]))
        all_children.add(j["child"])
    all_links = set(j["parent"] for j in joints.values()) | all_children
    roots = all_links - all_children
    return dict(children), roots


def print_tree(node, children, joints_by_child, prefix="", is_last=True):
    connector = "└── " if is_last else "├── "
    joint_name = joints_by_child.get(node, "")
    joint_tag = f"  [{joint_name}]" if joint_name else "  [ROOT]"
    print(f"{prefix}{connector}{node}{joint_tag}")
    child_prefix = prefix + ("    " if is_last else "│   ")
    kids = children.get(node, [])
    for i, (child, jname) in enumerate(kids):
        print_tree(child, children, joints_by_child, child_prefix, i == len(kids) - 1)


def main():
    links, joints = parse_urdf(URDF_PATH)

    # ── 1. Kinematic Tree ──────────────────────────────────────────────────────
    print("=" * 70)
    print("KINEMATIC TREE")
    print("=" * 70)
    children_map, roots = build_tree(joints)
    joints_by_child = {j["child"]: j["name"] for j in joints.values()}
    for root in sorted(roots):
        print_tree(root, children_map, joints_by_child)

    # ── 2. Joint Parameters ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"JOINTS  ({len(joints)} total)")
    print("=" * 70)
    print(f"{'Joint Name':<40} {'Type':<10} {'Axis':<12} {'Lower':>8} {'Upper':>8} {'Effort':>8} {'Vel':>6}")
    print("-" * 70)
    for j in joints.values():
        lim = j["limit"]
        if lim:
            lo, hi, eff, vel = lim["lower"], lim["upper"], lim["effort"], lim["velocity"]
        else:
            lo = hi = eff = vel = float("nan")
        print(f"{j['name']:<40} {j['type']:<10} {j['axis']:<12} {lo:>8.4f} {hi:>8.4f} {eff:>8.1f} {vel:>6.1f}")

    # ── 3. Link Inertial Properties ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"LINK INERTIAL PROPERTIES  ({len(links)} links)")
    print("=" * 70)
    total_mass = 0.0
    print(f"{'Link Name':<35} {'Mass (kg)':>10}  {'CoM (xyz)':<30}  {'ixx':>12} {'iyy':>12} {'izz':>12}")
    print("-" * 115)
    for link in links.values():
        m = link["mass"] or 0.0
        total_mass += m
        com = link["com"] or "-"
        if link["inertia"]:
            ixx = link["inertia"]["ixx"]
            iyy = link["inertia"]["iyy"]
            izz = link["inertia"]["izz"]
            print(f"{link['name']:<35} {m:>10.5f}  {com:<30}  {ixx:>12.6f} {iyy:>12.6f} {izz:>12.6f}")
        else:
            print(f"{link['name']:<35} {m:>10.5f}  {com:<30}  {'N/A':>12}")
    print("-" * 115)
    print(f"{'TOTAL MASS':<35} {total_mass:>10.4f} kg")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    joint_types = defaultdict(int)
    for j in joints.values():
        joint_types[j["type"]] += 1
    print(f"  Links        : {len(links)}")
    print(f"  Joints       : {len(joints)}")
    for jtype, count in sorted(joint_types.items()):
        print(f"    {jtype:<12}: {count}")
    print(f"  Total mass   : {total_mass:.4f} kg")


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
