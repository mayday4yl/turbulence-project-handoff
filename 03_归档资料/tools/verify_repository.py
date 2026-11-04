#!/usr/bin/env python3
"""Offline byte-integrity and payload-boundary check; Python standard library only."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile


MAX_FILE_BYTES = 50_000_000
PAYLOAD_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".h5", ".hdf5",
    ".hdf", ".mat", ".bin", ".pkl", ".pickle", ".onnx", ".pb", ".tflite",
    ".tar", ".gz", ".bz2", ".xz", ".zip", ".bundle", ".mp4", ".avi",
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError("unsafe relative path")
    return path


def check(root):
    errors = []
    inventory = {}
    byte_count = 0
    if root.is_symlink() or not root.is_dir():
        return {"status": "FAIL", "errors": ["root is not a regular directory"]}

    def walk_error(exc):
        errors.append("scan error: " + str(exc))

    for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
        for name in list(dirs):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                errors.append("symlink: " + rel)
                dirs.remove(name)
            elif rel == ".git":
                dirs.remove(name)
        for name in names:
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            try:
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    errors.append("symlink: " + rel)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    errors.append("non-regular file: " + rel)
                    continue
                if rel == ".git":  # Worktree administrative pointer, not payload.
                    continue
                inventory[rel] = info.st_size
                byte_count += info.st_size
                if info.st_size > MAX_FILE_BYTES:
                    errors.append("file exceeds 50000000 bytes: " + rel)
                    continue
                if path.suffix.lower() in PAYLOAD_SUFFIXES:
                    errors.append("excluded payload suffix: " + rel)
                    continue
                with path.open("rb") as stream:
                    prefix = stream.read(128)
                if prefix.startswith((b"\x93NUMPY", b"\x89HDF\r\n\x1a\n", b"MATLAB 5.0 MAT-file", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05")):
                    errors.append("array or serialized payload signature: " + rel)
                elif prefix.startswith(b"PK\x03\x04"):
                    with zipfile.ZipFile(path) as archive:
                        if any(PurePosixPath(n).name == "data.pkl" or PurePosixPath(n).suffix.lower() in {".npy", ".npz", ".pt", ".pth", ".safetensors"} for n in archive.namelist()):
                            errors.append("model or array archive: " + rel)
            except (OSError, zipfile.BadZipFile) as exc:
                errors.append("cannot inspect " + rel + ": " + str(exc))

    def read_json(rel):
        # Only scanned regular files are read, so no symlink is followed.
        if rel not in inventory or inventory[rel] > MAX_FILE_BYTES:
            raise ValueError("missing or invalid manifest: " + rel)
        return json.loads((root / rel).read_text(encoding="utf-8"))

    def verify_entries(entries, prefix, label):
        expected = set()
        verified = 0
        for value, record in entries:
            try:
                rel = relative_path(value).as_posix()
                if not rel.startswith(prefix):
                    raise ValueError("path outside " + prefix)
                if rel in expected:
                    raise ValueError("duplicate path")
                expected.add(rel)
                size, sha = record["bytes"], record["sha256"]
                if type(size) is not int or size < 0 or not isinstance(sha, str) or not SHA256.fullmatch(sha):
                    raise ValueError("invalid size/SHA metadata")
                if rel not in inventory:
                    raise ValueError("missing regular file")
                if inventory[rel] != size:
                    raise ValueError("byte count mismatch")
                if size > MAX_FILE_BYTES:
                    raise ValueError("file exceeds size boundary")
                if digest(root / rel) != sha:
                    raise ValueError("SHA256 mismatch")
                verified += 1
            except (KeyError, TypeError, ValueError, OSError) as exc:
                errors.append(label + " " + str(value) + ": " + str(exc))
        return expected, verified

    handoff_count = source_count = 0
    try:
        manifest = read_json("handoff/MANIFEST.json")
        if manifest.get("schema") != "portable_handoff_manifest_v1" or not isinstance(manifest.get("files"), dict):
            raise ValueError("unsupported handoff manifest schema")
        expected, handoff_count = verify_entries(
            (("handoff/" + str(k), v) for k, v in manifest["files"].items()),
            "handoff/", "handoff",
        )
        expected.add("handoff/MANIFEST.json")
        extra = {p for p in inventory if p.startswith("handoff/")} - expected
        errors.extend("unlisted handoff file: " + p for p in sorted(extra))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append("handoff manifest: " + str(exc))
    try:
        plan = read_json("provenance/SOURCE_SNAPSHOT_PLAN.json")
        if plan.get("schema") != "turbulence_source_snapshot_plan_v1" or not isinstance(plan.get("entries"), list):
            raise ValueError("unsupported source plan schema")
        expected, source_count = verify_entries(
            ((entry["repository_path"], entry) for entry in plan["entries"]),
            "source_snapshot/", "source",
        )
        if len(expected) != plan["counts"]["unique_files"]:
            errors.append("source plan unique-file count mismatch")
        if sum(entry["bytes"] for entry in plan["entries"]) != plan["counts"]["bytes"]:
            errors.append("source plan total byte count mismatch")
        extra = {p for p in inventory if p.startswith("source_snapshot/")} - expected
        errors.extend("unlisted source file: " + p for p in sorted(extra))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append("source plan: " + str(exc))

    return {
        "status": "FAIL" if errors else "PASS",
        "handoff_files_verified": handoff_count,
        "source_files_verified": source_count,
        "working_files_scanned": len(inventory),
        "working_bytes_scanned": byte_count,
        "max_file_bytes": MAX_FILE_BYTES,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).absolute().parent.parent)
    args = parser.parse_args()
    result = check(args.root.absolute())
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
