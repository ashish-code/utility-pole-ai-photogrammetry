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

import supervision as sv
from AI import AI
import cv2
import numpy as np
import os
from video import Video


class Diameter:
    def __init__(self):
        self.diameter_list = []
        self.median_diameter = self._find_median_diameter()
        self.ground_truth_diameter = 0.0

    def _find_median_diameter(self):
        if self.diameter_list == []:
            return 0.0
        else:
            return np.median(self.diameter_list)

    def compute_diameter(
        self,
        input_image_path: str,
        output_image_path: str,
        ai: AI,
        ground_truth_diameter: float = 0.0,
    ) -> None:
        """compute diameter of pole using visual detection and segmentation

        Args:
            input_image_path (str): path to input frame
            output_image_path (str): path to output frame
        """
        self.ground_truth_diameter = ground_truth_diameter
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

        # Initialize annotation text
        pole_width_text = ""

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
        badge_label_annotator = sv.LabelAnnotator(
            color=custom_color,
            text_color=sv.Color.BLACK,
            text_thickness=2,
            text_scale=1.5,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        mask_annotator = sv.MaskAnnotator(color=custom_color, opacity=0.3)

        # Process the detection classes separately
        detections_pole = detections[detections.class_id == 0]
        detections_badge = detections[detections.class_id == 1]

        # Skip the image/frame if no Pole is detected
        if len(detections_pole.box_area) == 0:
            # cv2.imwrite(output_image_path, annotated_image)
            return

        # ------------------------------------------------------------
        # Heuristics for pruning false-positive detections of Pole and Badge

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

        # -- Badge --

        # remove detections of badges that are not entirely contained within the detected pole
        if len(detections_badge.box_area) > 0:
            detections_badge = detections_badge[
                detections_badge.xyxy[:, 2] < detections_pole.xyxy[:, 2]
            ]
        if len(detections_badge.box_area) > 0:
            detections_badge = detections_badge[
                detections_badge.xyxy[:, 0] > detections_pole.xyxy[:, 0]
            ]
        if len(detections_badge.box_area) > 0:
            detections_badge = detections_badge[
                detections_badge.xyxy[:, 3] < detections_pole.xyxy[:, 3]
            ]
        if len(detections_badge.box_area) > 0:
            detections_badge = detections_badge[
                detections_badge.xyxy[:, 1] > detections_pole.xyxy[:, 1]
            ]

        # only select the biggest detected badge
        if len(detections_badge.box_area) > 0:
            biggest_badge_area = max(detections_badge.box_area)
            detections_badge = detections_badge[
                detections_badge.box_area >= biggest_badge_area
            ]

        # End of heuristics for pruning false-positive poles and badges
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
            pole_mask = pole_mask.tolist()

        if len(detections_badge.box_area) > 0:
            annotated_image = bounding_box_annotator.annotate(
                scene=annotated_image, detections=detections_badge
            )
            annotated_image = badge_label_annotator.annotate(
                scene=annotated_image, detections=detections_badge
            )

            # compute segmentation mask for BADGE
            # using Efficient-SAM
            detections_badge.mask = ai.inference_with_boxes(
                image=image,
                xyxy=detections_badge.xyxy,
                model=ai.segmentation_model,
                device=ai.device,
            )

            # annotate the image with the badge mask
            annotated_image = mask_annotator.annotate(
                scene=annotated_image, detections=detections_badge
            )

        # End of SEGMENTATION MASK
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # Computing Diameter

        # initialize badge-area / image_area
        badge_to_image_area = 0.0

        # compute diameter only if both valid pole and valid badge have been detected and selected
        if len(detections_pole.box_area) > 0 and len(detections_badge.box_area) > 0:
            detection_pole_xyxy = detections_pole.xyxy[0]
            detection_badge_xyxy = detections_badge.xyxy[0]

            # find the badge-area / image-area
            badge_to_image_area = detections_badge.box_area[0] / image_area

            # compute the vertical mid-point of the badge
            # the estimated diameter is of the virtual circle at the height of the mid-point of the BADGE
            badge_y_loc = int((detection_badge_xyxy[3] + detection_badge_xyxy[1]) // 2)

            # pole width/diameter
            pole_mask_y_loc = pole_mask[badge_y_loc]
            pole_x_min = pole_mask_y_loc.index(1)
            pole_x_max = len(pole_mask_y_loc) - 1 - pole_mask_y_loc[::-1].index(1)
            pole_width_y_loc = abs(pole_x_max - pole_x_min)
            pole_width_pixels = pole_width_y_loc

            # badge width/diameter
            badge_width_pixels = abs(detection_badge_xyxy[2] - detection_badge_xyxy[0])

            # compute the scaling factor due to depth discrepancy
            # scaling_factor = depth_scale_intercept + depth_scale_slope * 100 * (badge_to_image_area)
            scaling_factor = 1.32

            # real-world pole diameter
            pole_width_realworld = scaling_factor * (
                pole_width_pixels / badge_width_pixels
            )

            self.diameter_list.append(pole_width_realworld)
            self.median_diameter = self._find_median_diameter()

            # estimate error in mm
            error_estimate_inch = pole_width_realworld - ground_truth_diameter
            # estimate error in inches
            error_estimate_mm = 25.4 * error_estimate_inch

            # prune outlier pole diameter estimations
            if 4.0 < pole_width_realworld < 18.0:
                pole_width_text = (
                    f"Pole Diameter:"
                    + "\n"
                    + f"AI Estimation: {pole_width_realworld:.1f} inches"
                    + "\n"
                    + f"Ground Truth: {ground_truth_diameter:.1f} inches"
                )

        # Annotated image with pole diameter info
        if not pole_width_text == "":
            image_width = image.shape[1]
            image_height = image.shape[0]
            text_org_x = int(0.2 * image_width)
            text_org_y = int(0.85 * image_height)

            fontScale = 1.75
            text_thickness = 4
            text_padding = 10
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (10, 10, 10)
            # text_color_bg = (225, 225, 225)
            dia_color = (0, 255, 0)
            dia_thickness = 8
            title_text = "Pole Diameter"
            ai_text = f"A.I.: {pole_width_realworld:.2f} in, {25.4*pole_width_realworld:.2f} mm"
            manual_text = f"Manual: {ground_truth_diameter:.2f} in, {25.4*ground_truth_diameter:.2f} mm"
            error_text = (
                f"Error: {error_estimate_inch:.2f} in, {error_estimate_mm:.2f} mm"
            )

            (title_text_width, title_text_height), _ = cv2.getTextSize(
                title_text, font, fontScale, text_thickness
            )
            (ai_text_width, ai_text_height), _ = cv2.getTextSize(
                ai_text, font, fontScale, text_thickness
            )
            (manual_text_width, manual_text_height), _ = cv2.getTextSize(
                manual_text, font, fontScale, text_thickness
            )
            (error_text_width, error_text_height), _ = cv2.getTextSize(
                error_text, font, fontScale, text_thickness
            )

            if ground_truth_diameter == 0.0:
                text_width = ai_text_width
                text_height = ai_text_height
            else:
                text_width = max(
                    [
                        title_text_width,
                        ai_text_width,
                        manual_text_width,
                        error_text_width,
                    ]
                )
                text_height = max(
                    [
                        title_text_height,
                        ai_text_height,
                        manual_text_height,
                        error_text_height,
                    ]
                )

            text_org_1 = (text_org_x, text_org_y)
            text_org_2 = (text_org_x, text_org_y + text_height + text_padding)
            text_org_3 = (text_org_x, text_org_y + 2 * text_height + 2 * text_padding)
            text_org_4 = (text_org_x, text_org_y + 3 * text_height + 3 * text_padding)
            # annotated_image = cv2.rectangle(annotated_image, (text_org_x, text_org_y-text_height-text_padding), (int(0.9*image_width), int(0.95*image_height)), text_color_bg, -1)

            # Draw the translucent background for text annotation
            text_bg_x = text_org_x - 10
            text_bg_y = text_org_y - text_height - text_padding
            text_bg_w = text_width + 20

            if ground_truth_diameter == 0.0:
                text_bg_h = 2 * text_height + 3 * text_padding
            else:
                text_bg_h = 4 * text_height + 5 * text_padding

            text_bg_img = annotated_image[
                text_bg_y : text_bg_y + text_bg_h, text_bg_x : text_bg_x + text_bg_w
            ]
            text_bg_rect = np.ones(text_bg_img.shape, dtype=np.uint8) * 255
            text_bg = cv2.addWeighted(text_bg_img, 0.5, text_bg_rect, 0.5, 1.0)
            annotated_image[
                text_bg_y : text_bg_y + text_bg_h, text_bg_x : text_bg_x + text_bg_w
            ] = text_bg

            # Draw text to image
            annotated_image = cv2.putText(
                annotated_image,
                title_text,
                text_org_1,
                font,
                fontScale,
                text_color,
                text_thickness,
                cv2.LINE_AA,
                False,
            )
            annotated_image = cv2.putText(
                annotated_image,
                ai_text,
                text_org_2,
                font,
                fontScale,
                text_color,
                text_thickness,
                cv2.LINE_AA,
                False,
            )
            if ground_truth_diameter > 0.0:
                annotated_image = cv2.putText(
                    annotated_image,
                    manual_text,
                    text_org_3,
                    font,
                    fontScale,
                    text_color,
                    text_thickness,
                    cv2.LINE_AA,
                    False,
                )
                annotated_image = cv2.putText(
                    annotated_image,
                    error_text,
                    text_org_4,
                    font,
                    fontScale,
                    text_color,
                    text_thickness,
                    cv2.LINE_AA,
                    False,
                )

            # Draw the Diameter of the pole onto the image
            annotated_image = cv2.line(
                annotated_image,
                (pole_x_min, badge_y_loc),
                (pole_x_max, badge_y_loc),
                dia_color,
                dia_thickness,
            )

        # Save annotated image to file
        # cv2.imwrite(output_image_path, annotated_image)
        return annotated_image

    def compute_diameter_for_video(
        self, video: Video, ai: AI, ground_truth_diameter: float = 0.0
    ):
        input_image_path_list = [
            f
            for f in os.listdir(video.input_frames_directory_path)
            if f.endswith("jpg")
        ]

        for index, input_image_name in enumerate(input_image_path_list):
            output_image_name = input_image_name
            input_image_path = video.input_frames_directory_path + input_image_name
            output_image_path = video.output_frames_directory_path + output_image_name
            result_image = self.compute_diameter(
                input_image_path, output_image_path, ai, ground_truth_diameter
            )
            cv2.imwrite(output_image_path, result_image)
