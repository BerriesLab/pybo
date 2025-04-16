import os

import cv2

DEFAULT_OUTPUT_VIDEO_PATH = 'bo_1d_test.mp4'  # Centralized filename for the output video


def create_video_from_images(image_folder, output_video_path, fps=0.2, prefix=""):
    """
    Creates a video from a sequence of images in a folder.

    Args:
        image_folder (str): Path to the folder containing the images.
        output_video_path (str): Path to the output video file (e.g., 'output.mp4').
        fps (float, optional): Frames per second of the output video. Defaults to 0.2 to ensure slower playback.
        prefix (str, optional): Common prefix for image filenames to include. Defaults to an empty string.
    """
    images = [img for img in os.listdir(image_folder)
              if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')) and img.startswith(prefix)]
    images.sort()  # Sort the images to ensure correct order

    if not images:
        print(f"Error: No images found in the folder: {image_folder}")
        return

    # Read the first image to get dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4, other codecs may be needed for other video formats.
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        out.write(frame)  # Write the frame into the video

    # Release everything
    out.release()
    cv2.destroyAllWindows()
    print(f"Video successfully created at: {output_video_path}")


if __name__ == "__main__":
    # Example usage:
    image_folder = '../data'  # Replace with the path to your image folder
    output_video_path = DEFAULT_OUTPUT_VIDEO_PATH  # Use the centralized output filename
    fps = 1.0  # You can adjust the frames per second as needed

    # Create a dummy image folder and images for demonstration
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        for i in range(10):
            import numpy as np

            # Create a dummy image (black with a number on it)
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.putText(img, str(i), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(image_folder, f'image_{i:02d}.png'), img)
    create_video_from_images(image_folder, output_video_path, fps, prefix="test_1d_optimization - 2025-04-16_20-29-09")
