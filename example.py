import logging
import os
from sfm import estimate_trajectory

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filemode="w", filename="log.txt", format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    out_dir = "out_dir"
    os.makedirs(out_dir, exist_ok=True)
    estimate_trajectory("public_tests/00_test_slam_input", out_dir)