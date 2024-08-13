import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
import os


def load_model(model_path, device, num_classes=2):
    """Loads a pre-trained Faster R-CNN model and prepares it for inference.

    Args:
        model_path (str): Path to the saved model checkpoint file.
        device (torch.device): Device to use (CPU or GPU).
        num_classes (int, optional): Number of classes in the dataset (including background). Defaults to 2.

    Returns:
        torch.nn.Module: The loaded and prepared Faster R-CNN model.

    Raises:
        FileNotFoundError: If the model checkpoint file is not found.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = fasterrcnn_resnet50_fpn_v2(weights='COCO_V1')
    in_features = model.roi_heads.box_predictor.bbox_pred.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def transform_image(image):
    """Transforms an image for model input.

    Args:
        image (PIL.Image): The image to be transformed.

    Returns:
        torch.Tensor: The transformed image tensor with a batch dimension.
    """
    transform = T.Compose([
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def visualize_and_save_predictions(image, predictions, output_path, threshold=0.7):
    """Visualizes and saves predictions on an image.

    Args:
        image (PIL.Image): The original image.
        predictions (list): List containing model predictions (boxes, labels, scores).
        output_path (str): Path to save the visualized image with detections.
        threshold (float, optional): Confidence threshold to filter low-confidence detections. Defaults to 0.7.
    """
    image_np = np.array(image)
    boxes = predictions[0]['boxes'].cpu().detach().numpy()
    labels = predictions[0]['labels'].cpu().detach().numpy()
    scores = predictions[0]['scores'].cpu().detach().numpy()

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score < threshold:
            continue  # Skip detections below the confidence threshold
        x1, y1, x2, y2 = box
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image_np, f'Class {int(label)}: {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    output_image.save(output_path)
    print(f"Saved: {output_path}")


def test_model_on_image(model_path, image_path, output_path, device):
    """Tests the model on a single image and saves the output image with predictions.

    Args:
        model_path (str): Path to the saved model checkpoint.
        image_path (str): Path to the test image.
        output_path (str): Path to save the output image with predictions.
        device (torch.device): Device to use (CPU or GPU).
    """
    model = load_model(model_path, device)

    image = Image.open(image_path).convert('RGB')
    transformed_image = transform_image(image).to(device)

    with torch.no_grad():
        predictions = model(transformed_image)

    visualize_and_save_predictions(image, predictions, output_path)


if __name__ == "__main__":
    # Set your device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Path to the test image and output path
    image_path = "/data/malaria_proj/building_data/unification/dino/True_color_composite.png"
    output_path = "/data/malaria_proj/building_data/unification/dino/Ibadan.png"

    # Path to the saved model
    model_path = 'model_checkpoint.pth'

    # Run the testing
    test_model_on_image(model_path, image_path, output_path, device)

