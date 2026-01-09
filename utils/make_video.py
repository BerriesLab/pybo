import imageio.v2 as imageio
from pathlib import Path
import argparse


def make_video(filename, path=Path.cwd(), fps=1):
    path = Path(path)  # ensure it's a Path object
    image_files = sorted(path.glob(f"{filename}*.png"))
    if not image_files:
        raise ValueError("No images found matching the pattern.")

    # Output video path: same folder as images
    output_file = path / f"video_{filename}.mp4"

    with imageio.get_writer(
            output_file,
            fps=fps,
            codec="libx264",
            macro_block_size=1,  # allows any image dimensions
            quality=8
    ) as writer:
        for image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)

    print(f"Video saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an MP4 video from a sequence of PNG images."
    )
    parser.add_argument("filename", help="Common filename prefix (e.g. frame_)")
    parser.add_argument("--path", default=".", help="Directory containing images")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second")

    args = parser.parse_args()

    make_video(args.filename, Path(args.path), args.fps)
