#!/usr/bin/env python3
"""Extract one embedded resource from a NInfer v2 artifact, standard library only.

container.py is the full reader, but its import chain reaches layouts/numeric, which
pulls torch for tensor decoding that read-only resource extraction never needs. Use
this where torch is absent (e.g. the X99 runtime host); use inspect.py for the full
directory when the environment has the decode dependencies.

Also provides --dump-directory (raw JSON out) and --summary (object counts / byte
totals per kind) so artifact composition can be inspected without torch.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

MAGIC = b"NINFER\x00\x02"
PREFIX = struct.Struct("<8sQ")
PAYLOAD_ALIGNMENT = 4096


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("name", nargs="?", help="resource object name, e.g. frontend/tokenizer.json")
    parser.add_argument("output", nargs="?", type=Path, help="output path")
    parser.add_argument("--list", action="store_true", help="list resource names and exit")
    parser.add_argument("--dump-directory", type=Path,
                        help="write the raw directory JSON to this path and exit")
    parser.add_argument("--summary", action="store_true",
                        help="print per-kind object counts and byte totals, then exit")
    args = parser.parse_args()

    with args.artifact.open("rb") as handle:
        magic, json_bytes = PREFIX.unpack(handle.read(PREFIX.size))
        if magic != MAGIC:
            raise SystemExit(f"artifact magic is not NInfer v2: {magic!r}")
        directory = json.loads(handle.read(json_bytes).decode("utf-8"))
        payload_offset = _align_up(PREFIX.size + json_bytes, PAYLOAD_ALIGNMENT)

        if args.dump_directory is not None:
            args.dump_directory.write_text(json.dumps(directory, ensure_ascii=False),
                                           encoding="utf-8")
            print(f"directory JSON: {json_bytes} bytes -> {args.dump_directory}")
        if args.summary:
            for kind in ("tensor", "resource"):
                entries = [o for o in directory["objects"] if o.get("kind") == kind]
                total = sum(int(o["bytes"]) for o in entries)
                print(f"{kind:>8}: {len(entries):>5} objects, {total:>16} bytes")
            return
        if args.dump_directory is not None:
            return

        resources = [o for o in directory["objects"] if o.get("kind") == "resource"]
        if args.list or args.name is None:
            for entry in resources:
                print(f"{entry['bytes']:>12} {entry['encoding']:<14} {entry['name']}")
            print(f"{len(resources)} resources / {len(directory['objects']) - len(resources)} tensors")
            return
        if args.output is None:
            raise SystemExit("output path is required to extract")

        entry = next((o for o in resources if o["name"] == args.name), None)
        if entry is None:
            names = ", ".join(o["name"] for o in resources)
            raise SystemExit(f"resource not found: {args.name}\navailable: {names}")
        if entry["encoding"] != "raw-bytes-v1":
            raise SystemExit(f"unsupported resource encoding: {entry['encoding']}")
        handle.seek(payload_offset + int(entry["offset"]))
        payload = handle.read(int(entry["bytes"]))
        if len(payload) != int(entry["bytes"]):
            raise SystemExit("truncated resource payload")
        args.output.write_bytes(payload)
        print(f"{entry['name']}: {len(payload)} bytes -> {args.output}")


if __name__ == "__main__":
    main()
