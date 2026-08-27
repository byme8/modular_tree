import struct
import sys
import zlib
from pathlib import Path

MORPH_PREFIX = "Wind_"


def read_gltf(path):
    raw = path.read_bytes()

    if raw[:4] == b"glTF":
        length = struct.unpack_from("<I", raw, 8)[0]
        offset = 12
        chunk = None
        while offset < length:
            size, kind = struct.unpack_from("<II", raw, offset)
            if kind == 0x4E4F534A:
                chunk = raw[offset + 8 : offset + 8 + size]
                break
            offset += 12 + size - 4
        if chunk is None:
            return None
        import json

        document = json.loads(chunk.decode("utf-8"))
    else:
        import json

        document = json.loads(raw.decode("utf-8"))

    findings = []
    for mesh in document.get("meshes", []):
        names = mesh.get("extras", {}).get("targetNames", [])
        targets = sum(len(p.get("targets", [])) for p in mesh.get("primitives", []))
        findings.append(
            {
                "mesh": mesh.get("name", "<unnamed>"),
                "targets": targets,
                "names": names,
                "vertices": _gltf_vertex_count(document, mesh),
            }
        )

    return {
        "format": "glTF",
        "meshes": findings,
        "skins": len(document.get("skins", [])),
        "nodes": [n.get("name", "") for n in document.get("nodes", [])],
    }


def _gltf_vertex_count(document, mesh):
    total = 0
    for primitive in mesh.get("primitives", []):
        index = primitive.get("attributes", {}).get("POSITION")
        if index is None:
            continue
        total += document["accessors"][index].get("count", 0)
    return total


def read_fbx(path):
    raw = path.read_bytes()

    if raw[:21] != b"Kaydara FBX Binary  \x00":
        return read_fbx_ascii(raw)

    version = struct.unpack_from("<I", raw, 23)[0]
    wide = version >= 7500

    shapes = []
    deformers = 0
    models = []

    def walk(offset, depth):
        nonlocal deformers
        while True:
            if wide:
                end, count, prop_len = struct.unpack_from("<QQQ", raw, offset)
                header = 25
                name_len = raw[offset + 24]
            else:
                end, count, prop_len = struct.unpack_from("<III", raw, offset)
                header = 13
                name_len = raw[offset + 12]

            if end == 0:
                return offset + header

            name = raw[offset + header : offset + header + name_len].decode(
                "utf-8", "replace"
            )
            body = offset + header + name_len
            props_end = body + prop_len

            if name in ("Geometry", "Deformer", "Model"):
                text = raw[body:props_end]
                labels = _fbx_strings(text)
                if name == "Deformer":
                    if any("BlendShapeChannel" in s for s in labels):
                        shapes.extend(
                            s for s in labels if s and "BlendShapeChannel" not in s
                        )
                    if any("Skin" in s for s in labels):
                        deformers += 1
                if name == "Model":
                    models.extend(labels)

            cursor = props_end
            while cursor < end:
                cursor = walk(cursor, depth + 1)
            offset = end
            if offset >= len(raw) - 16:
                return offset

    try:
        walk(27, 0)
    except (struct.error, IndexError):
        pass

    return {
        "format": f"FBX binary v{version}",
        "shape_names": sorted(set(s for s in shapes if s.strip())),
        "skin_deformers": deformers,
        "models": sorted(set(m for m in models if m.strip()))[:20],
    }


def _fbx_strings(blob):
    found = []
    i = 0
    while i < len(blob) - 4:
        if blob[i : i + 1] == b"S":
            size = struct.unpack_from("<I", blob, i + 1)[0]
            if 0 < size < 200 and i + 5 + size <= len(blob):
                found.append(blob[i + 5 : i + 5 + size].decode("utf-8", "replace"))
                i += 5 + size
                continue
        i += 1
    return found


def read_fbx_ascii(raw):
    text = raw.decode("utf-8", "replace")
    shapes = []
    for line in text.splitlines():
        if "BlendShapeChannel" in line or MORPH_PREFIX in line:
            shapes.append(line.strip()[:120])
    return {
        "format": "FBX ascii",
        "shape_lines": shapes[:40],
        "skin_deformers": text.count('"Skin"'),
    }


def report(path):
    path = Path(path)
    if not path.exists():
        print(f"not found: {path}")
        return 1

    suffix = path.suffix.lower()
    if suffix in (".glb", ".gltf"):
        data = read_gltf(path)
    elif suffix == ".fbx":
        data = read_fbx(path)
    else:
        print(f"unsupported: {suffix}")
        return 1

    print(f"file    {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"format  {data['format']}")
    print()

    problems = []

    if "meshes" in data:
        for mesh in data["meshes"]:
            wind = [n for n in mesh["names"] if n.startswith(MORPH_PREFIX)]
            print(f"mesh    {mesh['mesh']}  {mesh['vertices']} verts")
            print(f"        morph targets: {mesh['targets']}")
            print(f"        wind morphs:   {len(wind)}  {wind}")
            if mesh["targets"] == 0:
                problems.append(f"{mesh['mesh']} carries no morph targets")
            elif not wind:
                problems.append(
                    f"{mesh['mesh']} has {mesh['targets']} targets but none named {MORPH_PREFIX}*"
                )
        print(f"skins   {data['skins']}")
        if data["skins"]:
            problems.append(
                f"{data['skins']} skin(s) exported; the armature should not travel"
            )
        armatures = [n for n in data["nodes"] if "Rig" in n or "Armature" in n]
        if armatures:
            problems.append(f"armature nodes present: {armatures}")

    if "shape_names" in data:
        wind = [s for s in data["shape_names"] if s.startswith(MORPH_PREFIX)]
        print(f"shapes  {len(data['shape_names'])} blend shape channel(s)")
        print(f"        wind morphs: {len(wind)}  {wind}")
        print(f"skins   {data['skin_deformers']} skin deformer(s)")
        print(f"models  {data['models']}")
        if not wind:
            problems.append(f"no blend shapes named {MORPH_PREFIX}*")
        if data["skin_deformers"]:
            problems.append(
                f"{data['skin_deformers']} skin deformer(s); the armature should not travel"
            )
        if any("Rig" in m or "Armature" in m for m in data["models"]):
            problems.append("armature model present in the file")

    if "shape_lines" in data:
        print(f"shape lines matched: {len(data['shape_lines'])}")
        for line in data["shape_lines"]:
            print(f"   {line}")

    print()
    if problems:
        print("PROBLEMS")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("OK: morphs present, no skeleton")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools/verify_morphs.py <file.fbx|file.glb|file.gltf>")
        sys.exit(2)
    sys.exit(max(report(p) for p in sys.argv[1:]))
