# import the inference-sdk
from inference_sdk import InferenceHTTPClient
from zipfile import ZipFile
import os
from os import listdir
import cv2

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="pOI9i83zbMXv4S4SJPzi"
)

# Path to the unzipped dataset
folder_dir = "/home/earthsense/Documents/unzipped_temp_dataset/temp_dataset"

# Check if the directory exists
if not os.path.exists(folder_dir):
    print(f"Directory {folder_dir} does not exist.")
    exit()

result_dict = {}

# Iterate through files and perform inference
for filename in os.listdir(folder_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(folder_dir, filename)
        print(f"Processing {filename}...")
        try:
            result = CLIENT.infer(image_path, model_id="visual-anomaly-detection/2")
            result_dict[filename] = result
            print(f"Result for {filename}: {result}")
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

        color = []
        if class_name == "mc4_connector":
            color = [0, 0, 255]
        else:
            color = [0, 255, 0]

        # if x and y start in the middle
        top_left_x = int(x - (width / 2))
        top_left_y = int(y - (height / 2))
        bottom_right_x = int(x + (width / 2))
        bottom_right_y = int(y + (height / 2))

        top_left = (top_left_x, top_left_y)
        bottom_right = (bottom_right_x, bottom_right_y)

        # label with class name and confidence
        label = f"{class_name}: {confidence:.2f}"
        label_pos = (top_left_x, top_left_y - 10)

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
