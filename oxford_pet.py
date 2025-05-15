import os
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import box_iou


class OxfordPetDetectionDataset(Dataset):
    def __init__(self, root, split="train", transforms=None, max_size=None, seed=42):
        self.root = os.path.join(root, "data")
        self.transforms = transforms
        self.split = split

        with open(os.path.join(self.root, "annotations", "list.txt")) as f:
            lines = f.readlines()[6:]  # skip header

        entries = []
        for line in lines:
            parts = line.strip().split()
            base_name = parts[0]
            species = int(parts[2])  # 1 = cat, 2 = dog
            label = 0 if species == 1 else 1  # 0 = cat, 1 = dog
            entries.append((base_name, label))

        if max_size:
            entries = entries[:max_size]

        # Split data
        train_val, test = train_test_split(entries, test_size=0.1, random_state=seed, stratify=[l for _, l in entries])
        train, val = train_test_split(train_val, test_size=0.15, random_state=seed, stratify=[l for _, l in train_val])

        if split == "train":
            self.data = train
        elif split == "val":
            self.data = val
        elif split == "test":
            self.data = test
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        base_name, label = self.data[idx]
    
        img_path = os.path.join(self.root, "images", base_name + ".jpg")
        mask_path = os.path.join(self.root, "annotations", "trimaps", base_name + ".png")
    
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 8-bit grayscale
    
        # Resize both image and mask to same size BEFORE extracting bounding boxes
        if self.transforms:
            image = self.transforms(image)
            mask = T.Resize((image.shape[1], image.shape[2]))(mask)  # match resized H/W
    
        mask = np.array(mask)
        obj_mask = (mask == 1)
    
        pos = np.where(obj_mask)
        # Box creation (always shape [1, 4] or [0, 4])
        if pos[0].size > 0 and pos[1].size > 0:
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)  # shape [1, 4]
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)  # shape [0, 4]
        
        target = {
            "boxes": boxes,  # already correct shape
            "labels": torch.ones((boxes.shape[0],), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "binary_label": label
        }
    
        return image, label, target


def plot_image_with_boxes(image, target):
    # Convert tensor image to numpy for plotting
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).numpy()
    else:
        image_np = image

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image_np)
    ax.axis("off")

    boxes = target['boxes']
    label = target.get('binary_label', None)

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

    # Optional title
    if label is not None:
        label_str = "Dog" if label == 1 else "Cat"
        ax.set_title(f"Label: {label_str}", fontsize=14)

    plt.show()

def evaluate_detr(model, processor, dataloader, device, iou_threshold=0.5, score_threshold=0.5, max_samples=None):
    model.eval()
    model.to(device)

    num_images = 0
    num_correct = 0
    ious = []
    total_gt = 0
    total_pred = 0
    total_true_positive = 0

    with torch.no_grad():
        for image, _, target in dataloader:
            img = image[0]

            # Make sure it's a PIL image
            if not isinstance(img, Image.Image):
                raise ValueError(f"Expected PIL.Image, got {type(img)}")

            inputs = processor(images=img, return_tensors="pt").to(device)
            width, height = img.size
            target_size = torch.tensor([[height, width]], device=device)
            outputs = model(**inputs)

            results = processor.post_process_object_detection(
                outputs, target_sizes=target_size, threshold=score_threshold
            )[0]

            pred_boxes = results["boxes"].to(device)
            scores = results["scores"]
            gt_boxes = target[0]["boxes"].to(device)

            # Count GT and predictions
            total_gt += len(gt_boxes)
            total_pred += len(pred_boxes)

            # Skip invalid or empty cases
            if pred_boxes.ndim != 2 or pred_boxes.shape[1] != 4:
                continue
            if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 4:
                continue
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue

            # Compute IoU matrix [num_pred, num_gt]
            iou_matrix = box_iou(pred_boxes, gt_boxes)

            # Best IoU for any pred/GT pair
            best_iou = iou_matrix.max().item()
            ious.append(best_iou)

            # For accuracy: count if any prediction matches GT above threshold
            if best_iou >= iou_threshold:
                num_correct += 1

            # For precision/recall: count true positives
            # A prediction is a true positive if it matches any GT with IoU >= threshold
            # For precision/recall: greedy matching (1 prediction per GT)
            matched_gt = set()
            true_positives = 0
            for pred_idx in range(iou_matrix.shape[0]):
                for gt_idx in range(iou_matrix.shape[1]):
                    if gt_idx in matched_gt:
                        continue
                    if iou_matrix[pred_idx, gt_idx] >= iou_threshold:
                        true_positives += 1
                        matched_gt.add(gt_idx)
                        break  # move to next prediction
            total_true_positive += true_positives


            num_images += 1
            if max_samples and num_images >= max_samples:
                break

            # Cleanup
            del inputs, outputs, pred_boxes, scores, gt_boxes, iou_matrix
            torch.cuda.empty_cache()

    if num_images == 0:
        print("⚠️ No valid samples with predictions and ground truth boxes.")
        return

    avg_iou = np.mean(ious)
    acc = num_correct / num_images
    precision = total_true_positive / total_pred if total_pred > 0 else 0.0
    recall = total_true_positive / total_gt if total_gt > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    print(f"\n📊 Evaluation Results (score_threshold={score_threshold}):")
    print(f"Average IoU: {avg_iou:.3f}")
    print(f"Detection Accuracy (IoU > {iou_threshold}): {num_correct}/{num_images} ({acc*100:.1f}%)")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")



def plot_detr_prediction(image, model, processor, device, target=None, score_threshold=0.7):
    model.eval()

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        width, height = image.size
        target_sizes = torch.tensor([[height, width]], device=device)

        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=score_threshold
        )[0]

    img_np = np.array(image)  # PIL -> numpy for plotting

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img_np)

    # Plot predicted boxes in lime green
    for box, score in zip(results["boxes"], results["scores"]):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f"{score:.2f}", color="lime", fontsize=8)

    # Plot ground truth boxes in red
    if target is not None and "boxes" in target and len(target["boxes"]) > 0:
        for box in target["boxes"]:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor="red", facecolor="none", linestyle="--")
            ax.add_patch(rect)
            ax.text(xmin, ymax + 5, "GT", color="red", fontsize=8)

    ax.set_title("DETR Prediction (green) + Ground Truth (red)")
    ax.axis("off")
    plt.show()
