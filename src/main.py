"""Osmose Project
Utility Pole Diameter Estimation

Objective: Estimate diameter of a utility with accuracy. 
Data: The data available is an image or video.
Approach: Use A.I. based methods. This project utilizes visual detection and visual object instance segmentation.
The visual detection model is YOLO-World, an open vocabulary visual object detection model.
The visual segmentation model is Efficient-SAM, an open vocabulary image segmentation model for instance segmentation.

Author: Ashish Gupta
Email: ashish@bright.ai
Date: 2024/07/03
"""

import argparse
from video import Video
from AI import AI
from diameter import Diameter
import time
from tilt import Tilt
import os


def pole_diameter():
    parser = argparse.ArgumentParser(
        description="Osmose Project: Utility Pole Diameter Estimation"
        "You can specify the input video file path with -i/--input, and output video file path with -o/--output."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="path (posix) to the input video file",
        default="data/sample/sample_1.mp4",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path (posix) to the output video file",
        default="",
    )
    parser.add_argument(
        "-d",
        "--diameter",
        type=float,
        help="Manual measurement of diameter of pole (inches)",
        default=0.0,
    )
    args = parser.parse_args()
    input_video_path = str(args.input)
    output_video_path = str(args.output)
    ground_truth_diameter = float(args.diameter)

    t_start = time.time()

    video = Video(input_video_path, output_video_path)
    video.extract_frames_from_video()

    ai = AI()

    diameter = Diameter()
    diameter.compute_diameter_for_video(video, ai, ground_truth_diameter)

    output_video_path = video.generate_video_from_frames()

    t_stop = time.time()
    t_elapsed = int(t_stop - t_start)
    print(
        f"Completed generating result video: {output_video_path} in {t_elapsed} seconds."
    )


def pole_tilt():
    parser = argparse.ArgumentParser(
        description="Osmose Project: Utility Pole Tilt Estimation"
        "You can specify the input image file path with -i/--input, and output image file path with -o/--output."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="path (posix) to the input image file",
        default="data/sample/pole_tilt/28490_GRANDVIEW_12589_Customer Required - Full Photo_Customer Required - Issue Photo_Leaning Pole - Major.jpg",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path (posix) to the output image file",
        default="result/pole_tilt/28490_GRANDVIEW_12589_Customer Required - Full Photo_Customer Required - Issue Photo_Leaning Pole - Major.jpg",
    )
    parser.add_argument(
        "-t",
        "--tilt",
        type=float,
        help="Manual measurement of tilt of pole (degrees)",
        default=0.0,
    )
    
    args = parser.parse_args()
    input_image_path = str(args.input)
    output_image_path = str(args.output)
    ground_truth_tilt = float(args.tilt)
    
    ai = AI()
    
    tilt = Tilt()
    tilt.compute_tilt(input_image_path, output_image_path, ai, ground_truth_tilt)
    
    print(f"Pole tilt: {tilt.tilt_angle} degrees")


def test_pole_tilt():
    input_image_dir = 'data/sample/pole_tilt/'
    output_image_dir = 'result/pole_tilt/'
    
    ai = AI()
    
    for image_name in os.listdir(input_image_dir):
        if image_name.endswith('jpg'):
            input_image_path = input_image_dir + image_name
            output_image_path = output_image_dir + image_name
            grouth_truth_tilt = 0.0
            tilt = Tilt()
            tilt.compute_tilt(input_image_path, output_image_path, ai, grouth_truth_tilt)
            print(f"Pole tilt: {tilt.tilt_angle} degrees")
            

if __name__ == "__main__":
    # pole_diameter()
    # pole_tilt()
    test_pole_tilt()
