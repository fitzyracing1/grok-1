#!/usr/bin/env python3
"""Estimate whether this machine can host Grok-1 weights at common precisions."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass


PARAM_COUNT = 314_000_000_000


@dataclass(frozen=True)
class PrecisionEstimate:
    name: str
    bits_per_param: int

    @property
    def bytes_required(self) -> int:
        return PARAM_COUNT * self.bits_per_param // 8


def format_bytes(num_bytes: int) -> str:
    gib = num_bytes / (1024 ** 3)
    tib = num_bytes / (1024 ** 4)
    if tib >= 1.0:
        return f"{tib:.2f} TiB ({gib:.1f} GiB)"
    return f"{gib:.1f} GiB"


def get_total_ram_bytes() -> int | None:
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(out)
        except Exception:
            return None
    return None


def main() -> None:
    estimates = [
        PrecisionEstimate("bf16/fp16", 16),
        PrecisionEstimate("8-bit", 8),
        PrecisionEstimate("4-bit", 4),
    ]

    ram = get_total_ram_bytes()
    disk = shutil.disk_usage(".").free

    print("Grok-1 System Feasibility Check")
    print("=" * 32)
    print(f"Platform: {platform.platform()}")
    if ram is not None:
        print(f"Total RAM: {format_bytes(ram)}")
    else:
        print("Total RAM: unknown")
    print(f"Free disk in current filesystem: {format_bytes(disk)}")
    print()

    print("Estimated raw weight memory requirements:")
    for item in estimates:
        req = item.bytes_required
        fit_ram = (ram is not None and req <= ram)
        fit_disk = req <= disk
        print(f"- {item.name:9s}: {format_bytes(req)} | RAM fit: {fit_ram} | Disk fit: {fit_disk}")

    print()
    print("Interpretation:")
    print("- These numbers are raw weights only and exclude runtime overhead, KV cache, and framework memory.")
    print("- Practical deployment needs additional headroom beyond these estimates.")
    print("- If RAM fit is false at your chosen precision, use remote multi-GPU infrastructure.")


if __name__ == "__main__":
    main()