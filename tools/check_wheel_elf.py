"""Fail-closed ELF machine check for packaged wheels.

Rejects any host-arch .so relabeled into a foreign-tagged wheel (e.g. x86_64
binaries inside a linux_aarch64 wheel), and any extension whose CPython ABI
tag names a different architecture than the wheel targets. Usage:

    python tools/check_wheel_elf.py <wheel> <expected-machine>

Exits nonzero unless every .so member is a valid 64-bit little-endian ELF
whose machine matches the expected one exactly, and no member's filename
carries a foreign architecture tag.
"""

import argparse
import sys
import zipfile

_MACHINES = {62: "X86-64", 183: "AArch64"}
_ELF_CLASS_64 = 2
_ELF_DATA_LITTLE = 1
# CPython encodes the platform in an extension's suffix
# ("_C.cpython-312-aarch64-linux-gnu.so"). The importer matches that suffix
# against the running interpreter, so a correctly-linked target binary saved
# under a host tag is silently unimportable on the target. Cross builds hit
# this when a build step derives EXT_SUFFIX from the build interpreter.
_NAME_TAGS = {"aarch64": "AArch64", "x86_64": "X86-64"}


def machine_of(data: bytes) -> str:
    if len(data) < 20 or data[:4] != b"\x7fELF":
        return "not-elf"
    if data[4] != _ELF_CLASS_64 or data[5] != _ELF_DATA_LITTLE:
        return f"unexpected-class-data({data[4]},{data[5]})"
    e_machine = int.from_bytes(data[18:20], "little")
    return _MACHINES.get(e_machine, f"unknown({e_machine})")


def _is_so(name: str) -> bool:
    return name.endswith(".so") or ".so." in name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel")
    parser.add_argument("expected", help="expected ELF machine, e.g. AArch64")
    args = parser.parse_args()

    expected = args.expected.lower().replace("-", "").replace("_", "")
    checked = 0
    try:
        zf = zipfile.ZipFile(args.wheel)
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"❌ cannot open {args.wheel}: {exc}", file=sys.stderr)
        return 1
    with zf:
        for name in zf.namelist():
            if not _is_so(name):
                continue
            machine = machine_of(zf.read(name))
            actual = machine.lower().replace("-", "").replace("_", "")
            if actual != expected:
                print(
                    f"❌ {name}: ELF machine {machine}, expected {args.expected}",
                    file=sys.stderr,
                )
                return 1
            for tag, tag_machine in _NAME_TAGS.items():
                if tag not in name.rsplit("/", 1)[-1]:
                    continue
                if tag_machine.lower().replace("-", "") != expected:
                    print(
                        f"❌ {name}: filename ABI tag names {tag_machine}, "
                        f"expected {args.expected}; CPython will not import "
                        f"this extension on the target",
                        file=sys.stderr,
                    )
                    return 1
            checked += 1

    if checked == 0:
        print(f"❌ no .so found in {args.wheel}", file=sys.stderr)
        return 1
    print(f"✅ {checked} shared objects in {args.wheel} are {args.expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
