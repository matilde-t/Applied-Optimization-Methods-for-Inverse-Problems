from pathlib import Path
import argparse

import utils
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(
        description="Simple example to load the HTC dataset and view one of the sinograms"
    )

    parser.add_argument("dir", type=Path, help="Path to extracted dataset")
    parser.add_argument(
        "--difficulty",
        type=int,
        default=1,
        help="Select the difficulty level of the phantom (1-7)",
    )
    parser.add_argument(
        "--phantom",
        type=str,
        default="a",
        help="Select the phantom of the given difficulty level (a-c)",
    )
    parser.add_argument(
        "--arc",
        type=int,
        default=360,
        help="Select the arc range of the reconstruction",
    )
    parser.add_argument(
        "--arc-start",
        type=int,
        default=0,
        help="Select the start of the arc (specifically for limited reconstructions)",
    )

    args = parser.parse_args()

    # Load the example reconstruction of the dataset
    gt_filename = Path(
        args.dir
        / f"htc2022_{str(args.difficulty).zfill(2)}{args.phantom}_recon_fbp.mat"
    )
    groundtruth = utils.loadmat(gt_filename)["reconFullFbp"].astype("float32")

    filename = Path(
        args.dir / f"htc2022_{str(args.difficulty).zfill(2)}{args.phantom}_full.mat"
    )

    # Setup and load the sinogram plus the X-ray projector/operator
    sinogram, A = utils.load_htc2022data(
        filename, arc=args.arc, arcstart=args.arc_start
    )

    # Do the back projection
    backprojection = (A.H * sinogram.flatten()).reshape((512, 512))

    # compute score comparted to groundtruth
    score = utils.calculate_score(
        utils.segment(backprojection), utils.segment(groundtruth)
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Show sinogram
    ax1.imshow(sinogram, cmap="gray")
    ax1.set_title(
        f"Sinogram of phantom {args.phantom} with difficulty {args.difficulty}",
    )

    # Show back projection
    ax2.imshow(backprojection, cmap="gray")
    ax2.set_title(
        f"Backprojection of Sinogram with score: {score}",
    )

    # Show ground truth
    ax3.imshow(groundtruth, cmap="gray")
    ax3.set_title(
        f"Groundtruth",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
