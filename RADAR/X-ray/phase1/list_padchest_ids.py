"""
List the PNG filenames contained in our PA/AP-only PadChest tar, without
extracting anything. Used as the join key against CheXmask's `ImageID`
column (see check_coverage.py / extract_masks.py).

Output: data/X-ray/PadChest-Origin/processed_chexmask/padchest_pa_ap_ids.txt
(one filename per line, e.g. "216840111366964012...5091.png").
"""
import os
import tarfile

TAR_PATH = "/datadrive/VLM/data/X-ray/PadChest-Origin/PNG_tar/PADCHEST_PA_AP.tar"
OUT_DIR = "/datadrive/VLM/data/X-ray/PadChest-Origin/processed_chexmask"
OUT_PATH = os.path.join(OUT_DIR, "padchest_pa_ap_ids.txt")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with tarfile.open(TAR_PATH) as tf:
        names = [os.path.basename(n) for n in tf.getnames() if n.lower().endswith(".png")]

    print(f"{len(names)} PNGs found in {TAR_PATH}")
    dupes = len(names) - len(set(names))
    if dupes:
        print(f"WARNING: {dupes} duplicate basenames")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(sorted(names)) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
