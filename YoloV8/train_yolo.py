from ultralytics import YOLO

# Path to your custom dataset YAML configuration file
yaml_path = 'custom.yaml'

# Create a YOLOv8 model instance
model = YOLO('yolov8m.pt')  # You can use 'yolov8n.yaml' for a small model or other variants like 'yolov8s.yaml', 'yolov8m.yaml', 'yolov8l.yaml', or 'yolov8x.yaml'

# Train the model
model.train(
    data=yaml_path,       # Path to the dataset YAML file
    epochs=75,            # Number of epochs to train
    device='0,1'
    
)

# Save the trained model
model.save('yolov8_custom_model.pt')

