import cv2
from pathlib import Path

DEFAULT_OUTPUT_VIDEO_NAME = 'movie.mp4'


def create_video_from_images(input_folder=None, output_folder=None, filename=None, fps=0.5):
    """
    Creates a video from a sequence of images stored in a specified folder using the
    OpenCV library. The output video is saved in the specified location with the
    desired filename and frame per second (fps) value.

    :param input_folder: A string specifying the folder containing image files to be
        used for video creation. Supports formats such as '.png', '.jpg', '.jpeg',
        and '.gif'.
    :param output_folder: A string specifying the folder where the generated video
        will be saved. If not provided, defaults to the `input_folder`.
    :param filename: A string specifying the name of the output video file.
        If not provided, the default filename is used.
    :param fps: A float specifying the frames per second (fps) for the generated
        video. Determines the playback speed of the video.
    :return: Returns a boolean value. `True` if the video is successfully created,
        otherwise `False`.
    """

    # Set folders and filename
    if input_folder is None:
        input_folder = Path.cwd()
    else:
        input_folder = Path(input_folder)

    if output_folder is None:
        output_folder = input_folder

    if filename is None:
        filename = DEFAULT_OUTPUT_VIDEO_NAME

    output_path = Path(output_folder) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all image files and sort them
    image_files = sorted([f for f in input_folder.glob('*') if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.gif')])

    if not image_files:
        print(f"Error: No images found in the folder: {input_folder}")
        return False

    # Read the first image to get dimensions
    frame = cv2.imread(str(image_files[0]))
    if frame is None:
        print(f"Error: Could not read image {image_files[0]}")
        return False

    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        for img_file in image_files:
            frame = cv2.imread(str(img_file))
            if frame is not None:
                out.write(frame)
            else:
                print(f"Warning: Could not read image {img_file}")

        print(f"Video successfully created at: {output_path}")
        return True

    except Exception as e:
        print(f"Error creating video: {str(e)}")
        return False

    finally:
        # Release everything
        out.release()
        cv2.destroyAllWindows()
