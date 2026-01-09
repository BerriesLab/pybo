import imageio
from pathlib import Path

# Get sorted list of images
image_files = sorted(Path("./frames").glob("*.png"))

# Create video
with imageio.get_writer("output.mp4", fps=1) as writer:
    for image_file in image_files:
        image = imageio.imread(image_file)
        writer.append_data(image)
