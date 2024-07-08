"""Osmose Project
Process the input and output video
Author: Ashish Gupta
Email: ashish@bright.ai
Date: 2024/07/03
"""

import os
import shutil
import subprocess


class Video:
    def __init__(self, input_video_path, output_video_path):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.output_video_dir = 'result/'

        self.input_frames_directory_path = "data/temp/input/"
        self.output_frames_directory_path = "data/temp/output/"

        if not os.path.exists(self.input_frames_directory_path):
            os.mkdir(self.input_frames_directory_path)
        else:
            self.clean_the_directory(self.input_frames_directory_path)

        if not os.path.exists(self.output_frames_directory_path):
            os.mkdir(self.output_frames_directory_path)
        else:
            self.clean_the_directory(self.output_frames_directory_path)

        if output_video_path == "":
            input_video_file_name = self.input_video_path.split('/')[-1]
            self.output_video_path = self.output_video_dir + input_video_file_name.split('.')[0] + '_result.mp4'
        
        output_video_dir = "/".join(output_video_path.split("/")[:-1])
        if not os.path.exists(output_video_dir):
            os.makedirs(self.output_video_dir, exist_ok=True)

        # Rate as which frames are extracted from video
        # Choose in [2 ... 30]
        self.frame_rate = 5

    def clean_the_directory(self, target_directory_path):
        for filename in os.listdir(target_directory_path):
            file_path = os.path.join(target_directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    def extract_frames_from_video(
        self,
    ) -> None:
        """extract image frames from video using ffmpeg

        Returns:
            None:
        """
        if not os.path.exists(self.input_video_path):
            print(
                "Video file not found. Please ensure input video file exists and try again."
            )
            return

        # call ffmpeg to extract frames from video
        ffmpeg_extract_cmd = f"ffmpeg -loglevel quiet -y -i {self.input_video_path} -r {self.frame_rate} {self.input_frames_directory_path}frame%04d.jpg"
        try:
            subprocess.call(ffmpeg_extract_cmd, shell=True)
        except Exception as e:
            print(f"Failed to extract frames from {self.input_video_path}. Reason: {e}")
            print(
                "Please install ffmpeg. On Mac you can use homebrew as brew install ffmpeg"
            )

    def generate_video_from_frames(self) -> str:
        """create a video from image frames using ffmpeg

        Returns:
            None
        """

        output_frames_list = [
            f
            for f in os.listdir(self.output_frames_directory_path)
            if f.endswith("jpg")
        ]

        if len(output_frames_list) == 0:
            print(
                "Error: Output frames not found. Cannot generate output video. \n Please update parameters and try again."
            )
            return

        ffmpeg_generate_video_cmd = f"ffmpeg -loglevel quiet -y -framerate {self.frame_rate} -i {self.output_frames_directory_path}frame%04d.jpg -c:v libx264 -r {self.frame_rate} -pix_fmt yuv420p {self.output_video_path}"
        try:
            subprocess.call(ffmpeg_generate_video_cmd, shell=True)
        except Exception as e:
            print(f"Failed to generate video {self.output_video_path}. Reason: {e}")
            print(
                "Please install ffmpeg. On Mac you can use homebrew as %brew install ffmpeg"
            )
        
        return self.output_video_path
