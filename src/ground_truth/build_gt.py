import argparse
import subprocess
import sys

MODULES = {
    "polynomial": "ground_truth.build_polynomial_gt",
    "gp": "ground_truth.build_gp_gt",
}


def main():
    parser = argparse.ArgumentParser(
        description="Fit a ground-truth surrogate for a collected pybo run's "
                    "objectives, either a polynomial regression or a Gaussian process.")
    parser.add_argument("model", choices=MODULES, help="Which model to fit")
    parser.add_argument("model_args", nargs=argparse.REMAINDER,
                        help="Forwarded to the chosen script, e.g. --degree 3 for "
                             "polynomial, or --kernel matern --nu 2.5 for gp")
    args = parser.parse_args()

    subprocess.run(
        [sys.executable, "-m", MODULES[args.model], *args.model_args], check=True
    )


if __name__ == "__main__":
    main()
