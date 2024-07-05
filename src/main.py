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
        default="result/sample_1_result.mp4",
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

    video.generate_video_from_frames()

    t_stop = time.time()
    t_elapsed = int(t_stop - t_start)
    print(
        f"Completed generating result video: {video.output_video_path} in {t_elapsed} seconds."
    )


if __name__ == "__main__":
    pole_diameter()
