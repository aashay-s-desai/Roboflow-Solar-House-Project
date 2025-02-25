# import the inference-sdk
from inference_sdk import InferenceHTTPClient

from zipfile import ZipFile

import os
from os import listdir

import cv2

#import numpy as np
#from pycocotools import mask as cocomask


# get the path/directory
folder_dir = "/home/earthsense/Documents/unzipped_temp_dataset"


# Dictionary to store results
result_dict = {}

# Check if the directory exists
if not os.path.exists(folder_dir):
    print(f"Directory {folder_dir} does not exist.")
else:
    print(f"Processing files in directory: {folder_dir}")

    # Iterate through files and perform inference
    for filename in os.listdir(folder_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_dir, filename)
            print(f"Processing {filename}...")
            try:
                result = CLIENT.infer(image_path, model_id="visual-anomaly-detection/2")
                result_dict[filename] = result
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Total files processed: {len(result_dict)}")

# Function to apply bounding boxes on images
def apply_annotations(image_path, predictions):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image {image_path}")
        return None

    for pred in predictions:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        class_name = pred['class']
        confidence = pred['confidence']
        
        # Set color based on class name
        color = [0, 0, 255] if class_name == "mc4_connector" else [0, 255, 0]

        # Calculate top-left and bottom-right points
        top_left = (int(x - (width / 2)), int(y - (height / 2)))
        bottom_right = (int(x + (width / 2)), int(y + (height / 2)))

        # Label with class name and confidence
        label = f"{class_name}: {confidence:.2f}"
        label_pos = (top_left[0], top_left[1] - 10)

        # Draw rectangle and put text on image
        cv2.rectangle(image, top_left, bottom_right, color, 2)
        cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# Apply annotations and save images
output_dir = "/home/earthsense/Documents/v2_annotated_images"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

for filename, result in result_dict.items():
    image_path = os.path.join(folder_dir, filename)
    predictions = result['predictions']
    annotated_image = apply_annotations(image_path, predictions)

    if annotated_image is not None:
        # Save annotated image
        annotated_image_path = os.path.join(output_dir, filename)
        cv2.imwrite(annotated_image_path, annotated_image)
        print(f"Saved annotated image to {annotated_image_path}")
    else:
        print(f"Failed to annotate {filename}")

print("Processing complete.")
