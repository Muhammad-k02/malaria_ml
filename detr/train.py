import os
import numpy as np
import torch
import torchvision.transforms as T
from monai.data import box_iou
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DetrForObjectDetection
from pycocotools.cocoeval import COCOeval
from torchvision.datasets import CocoDetection
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Dataset class
class CustomCocoDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(CustomCocoDataset, self).__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CustomCocoDataset, self).__getitem__(idx)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target


def evaluate(model, dataloader, coco):
    model.eval()
    total_loss = 0
    results = []
    predictions = []
    targets = []

    with torch.no_grad():
        for images, target in tqdm(dataloader, desc='Evaluating', leave=False):
            images = list(image.to(device) for image in images)

            # Prepare targets for the model
            annotations = []
            for tgt in target:
                boxes = torch.as_tensor([ann['bbox'] for ann in tgt], dtype=torch.float32).to(device)
                labels = torch.as_tensor([ann['category_id'] for ann in tgt], dtype=torch.int64).to(device)
                annotations.append({'boxes': boxes, 'labels': labels})

            # Model output
            outputs = model(images, annotations)
            loss = outputs.loss
            total_loss += loss.item()

            # Process predictions
            for i in range(len(images)):
                pred_boxes = outputs['pred_boxes'][i].detach().cpu().numpy()
                pred_logits = outputs['pred_logits'][i].detach().cpu().numpy()
                pred_scores = torch.nn.functional.softmax(torch.tensor(pred_logits), dim=-1).numpy()
                pred_labels = pred_logits.argmax(axis=-1)

                high_confidence = pred_scores.max(axis=-1) > 0.5
                pred_boxes = pred_boxes[high_confidence]
                pred_scores = pred_scores[high_confidence]
                pred_labels = pred_labels[high_confidence]

                image_id = target[i]['image_id']

                # Collect predictions
                predictions.append({
                    'boxes': pred_boxes,
                    'labels': pred_labels,
                    'scores': pred_scores
                })

                # Collect targets
                targets.append({
                    'boxes': [ann['bbox'] for ann in target[i]],
                    'labels': [ann['category_id'] for ann in target[i]]
                })

                # Format results for COCO evaluation
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    result = {
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": box.tolist(),
                        "score": score
                    }
                    results.append(result)

    # COCO evaluation
    coco_results = coco.loadRes(results)
    coco_eval = COCOeval(coco, coco_results, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    average_loss = total_loss / len(dataloader)
    logger.info(f'Evaluation Loss: {average_loss:.4f}')
    return average_loss, predictions, targets


def compute_metrics(predictions, targets):
    precision, recall, f1_score = [], [], []

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']

        target_boxes = target['boxes']
        target_labels = target['labels']

        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            if len(pred_boxes) == 0 and len(target_boxes) == 0:
                precision.append(1.0)
                recall.append(1.0)
                f1_score.append(1.0)
            else:
                precision.append(0.0)
                recall.append(0.0)
                f1_score.append(0.0)
            continue

        iou_matrix = box_iou(pred_boxes, target_boxes)

        tp = (iou_matrix > 0.5).sum().item()
        fp = len(pred_boxes) - tp
        fn = len(target_boxes) - tp

        if tp + fp > 0:
            precision.append(tp / (tp + fp))
        else:
            precision.append(0.0)

        if tp + fn > 0:
            recall.append(tp / (tp + fn))
        else:
            recall.append(0.0)

        if precision[-1] + recall[-1] > 0:
            f1_score.append(2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1]))
        else:
            f1_score.append(0.0)

    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f1_score = np.mean(f1_score)

    logger.info(f'Mean Precision: {mean_precision:.4f}')
    logger.info(f'Mean Recall: {mean_recall:.4f}')
    logger.info(f'Mean F1 Score: {mean_f1_score:.4f}')

    return mean_precision, mean_recall, mean_f1_score


transform = T.Compose([
    T.ColorJitter(brightness=(0.2, 0.5), contrast=(0.2, 0.5), saturation=(0.2, 0.5), hue=(0.1, 0.3)),
    T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    T.RandomGrayscale(p=0.2),
    T.RandomEqualize(p=0.1),
    T.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
    T.ToTensor(),
    T.ConvertImageDtype(torch.float32)
])


images_path = '/data/malaria_proj/building_data/unification/models/images'
annotations_path = 'coco_annotations.json'

full_dataset = CustomCocoDataset(images_path, annotations_path, transforms=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True, collate_fn=lambda x: tuple(zip(*x)),
                              pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=12, shuffle=False, collate_fn=lambda x: tuple(zip(*x)),
                            pin_memory=True)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=1, ignore_mismatched_sizes=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

num_epochs = 150

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in train_dataloader:
        images = list(image.to(device) for image in images)

        # Prepare targets
        annotations = []
        for target in targets:
            boxes = torch.as_tensor([ann['bbox'] for ann in target], dtype=torch.float32).to(device)
            labels = torch.as_tensor([ann['category_id'] for ann in target], dtype=torch.int64).to(device)
            annotations.append({'boxes': boxes, 'labels': labels})

        outputs = model(images, annotations)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    average_epoch_loss = epoch_loss / len(train_dataloader)
    logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_epoch_loss:.4f}')

    average_loss, predictions, targets = evaluate(model, val_dataloader, full_dataset.coco)
    precision, recall, f1_score = compute_metrics(predictions, targets)
    logger.info(f'Validation Loss after Epoch [{epoch + 1}/{num_epochs}]: {average_loss:.4f}')
    logger.info(f'Validation Precision: {precision:.4f}')
    logger.info(f'Validation Recall: {recall:.4f}')
    logger.info(f'Validation F1 Score: {f1_score:.4f}')

    scheduler.step(average_loss)

# Save the fine-tuned model
model_path = 'detr_finetuned.pth'
torch.save(model.state_dict(), model_path)
logger.info(f'Model saved to {model_path}')

