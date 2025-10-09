import base64
import json
import os
from typing import List, Dict
import numpy as np
import time
from typing import Union, List
import re
import PIL.Image
from io import BytesIO
import cv2
import argparse
from pathlib import Path
import shutil

# Retry decorator
def retry_api_call(max_attempts=3, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            last_exception = None
            while attempts < max_attempts:
                try:
                    result = func(*args, **kwargs)
                    time.sleep(delay)  # Pause for 2 seconds after each call
                    return result
                except Exception as e:
                    attempts += 1
                    last_exception = e
                    print(f"LLM call failed (attempt {attempts}/{max_attempts}): {str(e)}")
                    if attempts < max_attempts:
                        time.sleep(delay)
                    continue
            print(f"Maximum number of retries reached, the API call failed: {str(last_exception)}")
            return None  # Or return a default value depending on the function
        return wrapper
    return decorator

def read_txt_file(filename):
    """
    Read the content of a txt file.

    Args:
        filename (str): Path of the file.

    Returns:
        str: The whole content of the file, or None if reading fails.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: the file '{filename}' does not exist")
        return None
    except PermissionError:
        print(f"Error: no permission to read the file '{filename}'")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def copy_images(image_paths, target_folder):
    """
    Copy the images in the image path list into the target folder.

    Args:
        image_paths (list): List of image paths.
        target_folder (str): Path of the target folder.
    """
    # Create the target folder if it does not exist
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

    for img_path in image_paths:
        try:
            # Check whether the file exists
            if not os.path.exists(img_path):
                print(f"File does not exist: {img_path}")
                continue

            # Check whether it is an image file
            file_ext = os.path.splitext(img_path)[1].lower()
            if file_ext not in valid_extensions:
                print(f"Skip the non-image file: {img_path}")
                continue

            # Get the file name
            file_name = os.path.basename(img_path)

            # Build the target path
            target_path = os.path.join(target_folder, file_name)

            # Copy the file
            shutil.copy2(img_path, target_path)
            print(f"Copied: {img_path} -> {target_path}")

        except Exception as e:
            print(f"Error while copying {img_path}: {str(e)}")


def extract_frames_with_first_last(video_path, num_frames, output_dir='output/egtea'):
    """
    Extract the given number of evenly spaced frames from a video, making sure the first and the last frame are included.

    Args:
        video_path: Path of the video file.
        num_frames: Number of frames to extract.
        output_dir: Output directory.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video information
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Compute the indices of the frames to extract
    if num_frames == 1:
        # If only one frame is needed, take the first frame
        frame_indices = [0]
    elif num_frames == 2:
        # If two frames are needed, take the first and the last frame
        frame_indices = [0, total_frames - 1]
    else:
        # For more frames, make sure the first and the last frame are included
        # Pick evenly from the remaining frames
        frame_indices = [0]  # First frame
        step = (total_frames - 1) / (num_frames - 1)
        for i in range(1, num_frames - 1):
            frame_indices.append(int(i * step))
        frame_indices.append(total_frames - 1)  # Last frame

    frames_extracted = 0

    output = []

    for frame_index in frame_indices:
        # Set the current frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read the frame
        ret, frame = cap.read()

        # Save the frame
        filename = os.path.join(output_dir, f"frame_{frame_index:06d}.jpg")
        cv2.imwrite(filename, frame)
        output.append(filename)

    # Release the video capture object
    cap.release()

    # Return the number of extracted frames
    return output


def extract_and_load_json(json_string):
    """
    Extract the JSON part from a string and parse it with json.loads.
    
    Args:
    json_string: A string containing JSON (it may have a prefix and a suffix).
    
    Returns:
    The parsed Python object.
    """
    try:
        # Method 1: try to parse the whole string directly
        return json.loads(json_string)
    except json.JSONDecodeError:
        # Method 2: try to extract the JSON content between { and }
        # Find the content between the first { and the last }
        start = json_string.find('{')
        end = json_string.rfind('}') + 1
        
        if start >= 0 and end > start:
            try:
                return json.loads(json_string[start:end])
            except json.JSONDecodeError:
                return {}
        else:
            return {}

