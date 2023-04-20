from pathlib import Path
import argparse

from utils import load_htc2022data
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("dir", type=Path, help="Path to extracted dataset")
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--phantom", type=str, default="a")
    parser.add_argument("--arc", type=int, default=360)
    parser.add_argument("--arc-start", type=int, default=0)
    parser.add_argument(
        "--no-show", action="store_false", help="Do not show the final image"
    )
    parser.add_argument(
        "--save-to", type=Path, help="Path to save the reconstruction to as '.tif'"
    )

    args = parser.parse_args()

    filename = Path(
        args.dir / f"htc2022_{str(args.difficulty).zfill(2)}{args.phantom}_full.mat"
    )

    sinogram, A = load_htc2022data(filename, arc=args.arc, arcstart=args.arc_start)

    plt.imshow(sinogram, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
