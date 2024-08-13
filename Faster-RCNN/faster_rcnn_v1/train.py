import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor, Compose, Resize, RandomHorizontalFlip, ColorJitter, RandomRotation, \
    RandomAffine, GaussianBlur
import os
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import logging
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torchvision.ops import box_iou
from torch.optim.lr_scheduler import ReduceLROnPlateau


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def mask_to_boxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x + w, y + h])
    return np.array(boxes, dtype=np.float32)

class BuildingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png").replace(".jpeg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        boxes = mask_to_boxes(mask)

        if boxes.shape[0] == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)

        target = {"boxes": boxes, "labels": labels}

        return image, target

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

transform = Compose([
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ToTensor()
])

# Create the dataset
full_dataset = BuildingDataset(
    image_dir="/data/malaria_proj/building_data/unification/dino/images",
    mask_dir="/data/malaria_proj/building_data/unification/dino/masks",
    transform=transform
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True, collate_fn=collate_fn, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=12, shuffle=False, collate_fn=collate_fn, pin_memory=True)

model = fasterrcnn_resnet50_fpn_v2(weights='COCO_V1')
in_features = model.roi_heads.box_predictor.bbox_pred.in_features
num_classes = 2
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
scaler = GradScaler()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

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

    return np.mean(precision), np.mean(recall), np.mean(f1_score)

def evaluate_model(dataloader):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        with tqdm(dataloader, desc='Evaluating', unit='batch') as pbar:
            for images, targets in pbar:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = model(images)
                all_predictions.extend(predictions)
                all_targets.extend(targets)

                pbar.set_postfix({"Batch": pbar.n + 1})

    precision, recall, f1_score = compute_metrics(all_predictions, all_targets)
    logging.info(f" Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")

    return precision, recall, f1_score


num_epochs = 150

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for images, targets in pbar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with autocast():
                loss_dict = model(images, targets)
                logging.debug(f"Training loss dict: {loss_dict}")
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += losses.item()
            avg_loss = running_loss / (pbar.n + 1)
            pbar.set_postfix(loss=avg_loss)

    precision, recall, f1_score = evaluate_model(val_dataloader)
    scheduler.step(avg_loss)

    logging.info(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val F1-score: {f1_score:.4f}")
    
torch.save(model.state_dict(), 'model_checkpoint.pth')
logging.info("Training complete.")
