"""Osmose Project
Utility Pole Tilt Estimation

Objective: Estimate tilt of a utility with accuracy. 
Data: The data available is an image.
Approach: Use A.I. based methods. This project utilizes visual detection and visual object instance segmentation.
The visual detection model is YOLO-World, an open vocabulary visual object detection model.
The visual segmentation model is Efficient-SAM, an open vocabulary image segmentation model for instance segmentation.

Author: Ashish Gupta
Email: ashish@bright.ai
Date: 2024/07/18
"""


import supervision as sv
from AI import AI
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import skimage
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line

class Tilt:
    def __init__(self):
        self._tilt_angle = 0.0
    
    @property
    def tilt_angle(self):
        return self._tilt_angle

    @tilt_angle.setter
    def tilt_angle(self, tilt_angle):
        self._tilt_angle = tilt_angle
    
    def compute_tilt(self, 
                     input_image_path: str, 
                     output_image_path: str, 
                     ai: AI,
                     ground_truth_tilt: float):
        """compute angle (degree) of tilt of utility pole from vertical

        Args:
            input_image_path (str): path (posix) of the pole image
            ai (AI): object detection and segmentation
        """
        self.ground_truth_tilt = ground_truth_tilt
        custom_color = sv.ColorPalette.from_hex(["#1616f7", "#f71616"])
        
        # Read Image and Pre-process
        image_bgr = cv2.imread(input_image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Initialize image/frame with no viable detections
        annotated_image = image_bgr

        # CALL DETECTION MODEL
        results = ai.detection_model.predict(
            source=image,
            conf=ai.confidence_threshold,
            iou=ai.iou_threshold,
            max_det=ai.max_detections,
            verbose=False,
        )[0]

        # Parse detection results into Supervision object
        detections = sv.Detections.from_ultralytics(results)
        
        # Supervision library annotator objects
        bounding_box_annotator = sv.BoundingBoxAnnotator(
            thickness=6, color=custom_color
        )
        label_annotator = sv.LabelAnnotator(
            color=custom_color,
            text_color=sv.Color.BLACK,
            text_thickness=2,
            text_scale=1.5,
            text_position=sv.Position.CENTER_RIGHT,
        )
        mask_annotator = sv.MaskAnnotator(color=custom_color, opacity=0.3)
        
        detections_pole = detections[detections.class_id == 0]
        
        # Skip the image/frame if no Pole is detected
        if len(detections_pole.box_area) == 0:
            return
        
        # ------------------------------------------------------------
        # Heuristics for pruning false-positive detections of Pole

        image_size = tuple(image.shape[:2])
        image_height = image_size[0]
        image_width = image_size[1]
        image_area = image_height * image_width

        # -- POLE --
        box_w = detections_pole.xyxy[:, 2] - detections_pole.xyxy[:, 0]
        box_h = detections_pole.xyxy[:, 3] - detections_pole.xyxy[:, 1]

        # Only select detections where pole height is greater than pole width
        detections_pole = detections_pole[box_h > box_w]

        box_w = detections_pole.xyxy[:, 2] - detections_pole.xyxy[:, 0]
        box_h = detections_pole.xyxy[:, 3] - detections_pole.xyxy[:, 1]

        # only select detections where pole height is atleast 90% of the image height
        # detections_pole = detections_pole[box_h > 0.9 * image_height]

        # only select detections where pole area in pixels is at least 4% of the image area
        detections_pole = detections_pole[detections_pole.box_area > 0.04 * image_area]

        # only select detections where pole area is pixels does not exceed 75% of the image area
        max_pole_area = 0.75 * image_area
        detections_pole = detections_pole[detections_pole.box_area < max_pole_area]

        # only select the biggest detected pole
        if len(detections_pole.box_area) > 0:
            biggest_pole_area = max(detections_pole.box_area)
            detections_pole = detections_pole[
                detections_pole.box_area >= biggest_pole_area
            ]

        # if no valid detections remain, skip the image/frame from further processing
        if len(detections_pole.box_area) == 0:
            cv2.imwrite(output_image_path, annotated_image)
            return

        # ------------------------------------------------------------
        
        # ------------------------------------------------------------
        # SEGMENTATION MASK

        # initialize mask for POLE
        pole_mask = None

        if len(detections_pole.box_area) > 0:
            annotated_image = bounding_box_annotator.annotate(
                scene=annotated_image, detections=detections_pole
            )
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections_pole
            )

            # use the detected bounding-box to find segmentation mask for pole
            # using Efficient-SAM
            detections_pole.mask = ai.inference_with_boxes(
                image=image,
                xyxy=detections_pole.xyxy,
                model=ai.segmentation_model,
                device=ai.device,
            )

            # annotate the image with the pole mask
            annotated_image = mask_annotator.annotate(
                scene=annotated_image, detections=detections_pole
            )

            pole_mask = detections_pole.mask[0]
            medial_axis = skimage.morphology.medial_axis(pole_mask)
            
            medial_axis_image = 255 * medial_axis
            
            
            # straight-line Hough Transform
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
            h, theta, d = hough_line(medial_axis_image, theta=tested_angles)
            
            bestH, bestTheta, bestD = skimage.transform.hough_line_peaks(h, theta, d)
            
            angle = np.median(bestTheta * (180/np.pi))
            angle = np.round(angle, 2)
            self._tilt_angle = angle
            
            # print(f'Tilt angle: {angle} degrees')
            
            # DEBUG
            fig, axes = plt.subplots(2, 2, figsize=(8,8))
            image_name = input_image_path.split('/')[-1].split('.')[0]
            fig.suptitle(f'Utility Pole Tilt: {angle} degrees.\n{image_name}')
            axes[0, 0].imshow(image)
            axes[0, 0].axis('off')
            axes[0, 0].set_title('original image')
            
            axes[0, 1].imshow(pole_mask.view(np.uint8), cmap='bone')
            axes[0, 1].axis('off')
            axes[0, 1].set_title('pole segmentation')
            
            axes[1, 0].imshow(np.invert(medial_axis).view(np.uint8), cmap=cm.gray)
            # axes[1, 0].set_title('medial-axis-transform')
            axes[1, 0].set_title('thinned pole mask')
            
            axes[1, 1].imshow(image, cmap=cm.gray)
            axes[1, 1].set_ylim((medial_axis_image.shape[0], 0))
            axes[1, 1].set_axis_off()
            axes[1, 1].set_title('pole tilt candidate(s)')

            for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
                (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
                axes[1, 1].axline((x0, y0), slope=np.tan(angle + np.pi / 2), color='red', linestyle='-.')
            
            # save image to file
            plt.savefig(output_image_path)
            plt.show()
            
            
