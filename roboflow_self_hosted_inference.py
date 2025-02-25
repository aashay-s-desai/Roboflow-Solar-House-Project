import os
#import json
import ast
import subprocess
import cv2

# Directory containing your images
folder_dir = "/home/earthsense/Documents/shorter_unzipped_temp_dataset"

# Loop over each image in the directory
result_dict = {}
for filename in os.listdir(folder_dir):
    if filename.endswith(".jpg"):  # Add other formats if needed
        image_path = os.path.join(folder_dir, filename)
        command = [
            "inference", "infer",
            "-i", image_path,
            "-m", "visual-anomaly-detection/2",
            "--api-key", "pOI9i83zbMXv4S4SJPzi"
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        print(f"Processed {filename}: {result.stdout}")
        
        # Extract the JSON part of the output (In Colab, when you call the CLIENT.infer() method, you're directly interacting with the API, and the response is a pure JSON object without additional logging or process-related information. So, to compensate, we can extract just the JSON part of the output)
        print("NOW JSON SDLKFJSLDFJS:LDKJS:LDKFJ:SLDKFJSL:DFKJ:SLDFKJS:DLKFJS:DLKJ")
        output = result.stdout #.strip()???
        json_start = output.find("{")
        json_output = output[json_start:]
        
        try:
        
        	json_data = ast.literal_eval(json_output) #so that its in json format and when its tryna extract numbers and stuff its not all strings
        	print(json_data)
            
        	# Store the result
       		result_dict[filename] = json_data
            
        	print(f"Processed {filename}: {json_data}")
        
        except ValueError as e:
        	print(f"Failed to extract JSON from the output for {filename}: {e}")
            
print(result_dict)

# Now you can work with result_dict just like in Colab



#apply bounding boxes on images
def apply_annotations(image_path, predictions):
    image = cv2.imread(image_path)
    for pred in predictions:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        class_name = pred['class']
        confidence = pred['confidence']

        color = []
        if (class_name == "mc4_connector") :
          color = [0, 0, 255]
        else :
          color = [0, 255, 0]

        '''
        #if x and y start in the top left
        # Draw bounding box
        top_left = (int(x), int(y))
        bottom_right = (int(x + width), int(y + height))
        #cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
        '''
        #if x and y start in the middle
        top_left_x = int(x - (width / 2))
        top_left_y = int(y - (height / 2))
        bottom_right_x = int(x + (width / 2))
        bottom_right_y = int(y + (height / 2))

        top_left = (top_left_x, top_left_y)
        bottom_right = (bottom_right_x, bottom_right_y)

        #label with class name and confidence
        label = f"{class_name}: {confidence:.2f}"
        label_pos = (top_left_x, top_left_y - 10)


        cv2.rectangle(image, top_left, bottom_right, color, 2)
        cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image



#apply annotations and save/display images
for filename, result in result_dict.items():
    image_path = os.path.join(folder_dir, filename)
    predictions = result['predictions']
    annotated_image = apply_annotations(image_path, predictions)

    #save annotated image
    annotated_image_path = os.path.join("/home/earthsense/Documents/ANOTHER_new_v2_annotated_images", filename)
    os.makedirs(os.path.dirname(annotated_image_path), exist_ok=True)
    cv2.imwrite(annotated_image_path, annotated_image)

    #display the image:
    #cv2.imshow("Annotated Image", annotated_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
