import torch
import numpy as np

# import supervision as sv
from ultralytics import YOLOWorld
from torchvision.transforms import ToTensor
import subprocess
import os

EFFICIENT_SAM_CPU_URL = "https://huggingface.co/spaces/SkalskiP/YOLO-World/resolve/main/efficient_sam_s_cpu.jit"
EFFICIENT_SAM_GPU_URL = "https://huggingface.co/spaces/SkalskiP/YOLO-World/resolve/main/efficient_sam_s_gpu.jit"


class AI:
    def __init__(
        self,
    ):
        self.classes = ["POLE", "BADGE"]
        self.confidence_threshold = 0.0001
        self.iou_threshold = 0.4
        self.max_detections = 20
        self.depth_scale_param = 1.32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_world_model_path = "model/yolov8x-world.pt"
        self.gpu_efficient_sam_model_path = "efficient_sam_s_gpu.jit"
        self.cpu_efficient_sam_model_path = "efficient_sam_s_cpu.jit"
        # self.detection_model = YOLOWorld(self.yolo_world_model_path)
        self.detection_model = YOLOWorld("yolov8x-world.pt")
        self.detection_model.set_classes(self.classes)
        self.segmentation_model = self._load_SAM_model(self.device)

    def _load_SAM_model(self, device: torch.device) -> torch.jit.ScriptModule:
        if device.type == "cuda":
            if not os.path.exists(self.gpu_efficient_sam_model_path):
                download_model_cmd = f"wget {EFFICIENT_SAM_GPU_URL}"
                subprocess.call(download_model_cmd, shell=True)
            model = torch.jit.load(self.gpu_efficient_sam_model_path)
        else:
            if not os.path.exists(self.cpu_efficient_sam_model_path):
                download_model_cmd = f"wget {EFFICIENT_SAM_CPU_URL}"
                subprocess.call(download_model_cmd, shell=True)
            model = torch.jit.load(self.cpu_efficient_sam_model_path)

        model.eval()
        return model

    def inference_with_box(
        self,
        image: np.ndarray,
        box: np.ndarray,
        model: torch.jit.ScriptModule,
        device: torch.device,
    ) -> np.ndarray:
        bbox = torch.reshape(torch.tensor(box), [1, 1, 2, 2])
        bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
        img_tensor = ToTensor()(image)

        predicted_logits, predicted_iou = model(
            img_tensor[None, ...].to(device),
            bbox.to(device),
            bbox_labels.to(device),
        )
        predicted_logits = predicted_logits.cpu()
        all_masks = torch.ge(
            torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5
        ).numpy()
        predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

        max_predicted_iou = -1
        selected_mask_using_predicted_iou = None
        for m in range(all_masks.shape[0]):
            curr_predicted_iou = predicted_iou[m]
            if (
                curr_predicted_iou > max_predicted_iou
                or selected_mask_using_predicted_iou is None
            ):
                max_predicted_iou = curr_predicted_iou
                selected_mask_using_predicted_iou = all_masks[m]
        return selected_mask_using_predicted_iou

    def inference_with_boxes(
        self,
        image: np.ndarray,
        xyxy: np.ndarray,
        model: torch.jit.ScriptModule,
        device: torch.device,
    ) -> np.ndarray:
        masks = []
        for [x_min, y_min, x_max, y_max] in xyxy:
            box = np.array([[x_min, y_min], [x_max, y_max]])
            mask = self.inference_with_box(image, box, model, device)
            masks.append(mask)
        return np.array(masks)
