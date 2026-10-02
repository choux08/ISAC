"""Standalone: scan an att_cache_dir (produced by cache_teacher_attention.py) for corrupted
.pt files and delete them, so a re-run of cache_teacher_attention.py's resume-skip logic
(which only checks file existence, not validity) will regenerate just those entries.

Root cause this fixes (2026-09-29): a Ctrl+C / preempted instance mid-torch.save could leave
a truncated .pt file under the old (non-atomic) save path. cache_teacher_attention.py now
saves atomically, so this shouldn't recur going forward -- this script is a one-time cleanup
for cache directories built before that fix.

Usage:
    python verify_att_cache.py ~/att_cache/qwen25_7b
    python verify_att_cache.py ~/att_cache/qwen25_7b --dry-run   # report only, don't delete
"""
import argparse
import glob
import os

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("att_cache_dir")
    parser.add_argument("--dry-run", action="store_true", help="Report corrupted files without deleting them.")
    args = parser.parse_args()

    cache_dir = os.path.expanduser(args.att_cache_dir)
    pt_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    print(f"Checking {len(pt_files)} cached files under {cache_dir} ...")

    corrupted = []
    for i, path in enumerate(pt_files):
        try:
            torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            corrupted.append(path)
            print(f"  CORRUPTED: {os.path.basename(path)} ({e})")
        if (i + 1) % 20000 == 0:
            print(f"  ...checked {i + 1}/{len(pt_files)}")

    print(f"\n{len(corrupted)} corrupted file(s) found out of {len(pt_files)}.")
    if corrupted and not args.dry_run:
        for path in corrupted:
            os.remove(path)
        print(f"Deleted {len(corrupted)} corrupted file(s). Re-run cache_teacher_attention.py "
              "(or the parallel launcher) to regenerate just these -- the resume-skip logic "
              "will leave everything else untouched.")
    elif corrupted:
        print("--dry-run set: nothing deleted.")


if __name__ == "__main__":
    main()
