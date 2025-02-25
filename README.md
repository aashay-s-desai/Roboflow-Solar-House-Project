# Roboflow Self-Hosted Inference Project

This project is designed for running **visual anomaly detection** using Roboflow's self-hosted inference server. 
It processes images, performs inference, and applies bounding boxes to detected objects.

## 1. Cloning the Repository
Start by cloning the repository:

```bash
git clone https://github.com/YOUR_USERNAME/roboflow-inference-project.git
cd roboflow-inference-project
```

## 2. Setup Steps

### Install Dependencies
Ensure you have **Python 3.10+** installed and create a virtual environment:

```bash
conda create -n roboflow python=3.10 -y
conda activate roboflow
```

### Install Required Packages
```bash
pip install opencv-python-headless inference-sdk
```

### Install and Setup Roboflow Inference Server
To run inference locally, you need to install Roboflow's inference server:

```bash
pip install inference
```

---

## 3. Extract Input Dataset
This repository does not include the dataset ZIP file. **If you have a zipped dataset,** unzip it manually or add this functionality:

```python
from zipfile import ZipFile
import os

zip_path = "/path/to/dataset.zip"
extract_path = "/home/earthsense/Documents/unzipped_temp_dataset"

os.makedirs(extract_path, exist_ok=True)
with ZipFile(zip_path, 'r') as zipObj:
    zipObj.extractall(extract_path)
```

Ensure that the dataset is extracted into the correct directory before running inference.

---

## 4. Running Inference

### Using Cloud Inference (Roboflow API)
Run the following script:

```bash
python roboflow_script.py
```

This will:
- Load images from `/home/earthsense/Documents/unzipped_temp_dataset/temp_dataset`
- Perform inference using **Roboflow’s API**
- Apply bounding boxes to detected objects
- Save the annotated images to `/home/earthsense/Documents/v2_annotated_images`

---

### Using **Self-Hosted Inference Server**
Run the following script:

```bash
python roboflow_self_hosted_inference_.py
```

This script:
- Uses **Roboflow's self-hosted inference** instead of the cloud API.
- Processes images from `/home/earthsense/Documents/shorter_unzipped_temp_dataset`
- Saves annotated images to `/home/earthsense/Documents/ANOTHER_new_v2_annotated_images`

**Ensure that the inference server is running before executing the script.**

---

## 5. Outputs
- Cloud inference results (from `roboflow_script.py`) are stored in:
  ```
  /home/earthsense/Documents/v2_annotated_images
  ```
- Self-hosted inference results (from `roboflow_self_hosted_inference_.py`) are stored in:
  ```
  /home/earthsense/Documents/ANOTHER_new_v2_annotated_images
  ```

Both directories contain images with applied bounding boxes.

---

## 6. Notes
- If you need to **run inference on a different dataset**, update the `folder_dir` variable inside the script.
- To modify inference parameters (e.g., confidence thresholds), edit the relevant scripts before running.
- If you encounter errors related to missing API keys, ensure you have the correct `api_key` in `roboflow_script.py`.

---

## 7. Future Improvements
- Automate dataset downloading and extraction.
- Add support for batch inference.
- Improve error handling for API responses.

This project enables **both cloud-based and self-hosted inference**, giving flexibility for different use cases. 🚀
