# Advanced Visual Explainability Pipeline


## Overview

We have designed a simplified implementation pipeline based on our research paper: 'Towards Human-Understandable Explanations in Explainable AI Using Large Multimodal Models'. This pipeline uses a custom-trained YOLOv8 model, along with visual explainability techniques including SHAP (SHapley Additive exPlanations) and DRISE (Dynamic Randomized Input Sampling for Explanations). The pipeline processes images, generates bounding boxes, SHAP visualizations, saliency maps, and allows interaction with a large multimodal model (like Idefics2) to interpret these results in concise, human-level explanations. This document explains how the pipeline works and guides users through its execution.


### Key Features
- **YOLOv8**: Performs object detection, generating bounding boxes around identified objects.
- **SHAP (SHapley Additive exPlanations)**: Provides visual explanations by highlighting image regions that contribute positively or negatively to object detection.
- **DRISE (Dynamic Randomized Input Sampling for Explanations)**: Generates saliency maps highlighting areas most influential for the detection.
- **Idefics2 (Large Multimodal Model)**: Offers human-level interpretations of visual explanations via natural language interaction.


### Enhanced Features

- **Bounding Box Toggle:** Easily enable or disable bounding boxes on visualizations.
- **Custom Superpixeling:** The SHAP implementation now utilizes a customized SLIC segmentation method, significantly improving explainability precision.


### Prerequisites
Install dependencies with:
```bash
pip install transformers==4.49.0 ultralytics numpy opencv-python shap torch torchvision torchaudio flash-attn bitsandbytes accelerate
```

## Running the Pipeline

### Part 1: Object Detection with YOLOv8 and Explainability with SHAP and DRISE

#### 1. YOLOv8 Object Detection

The pipeline begins by loading a custom-trained YOLOv8 model, which is used for object detection on input images. The model generates prediction bounding boxes for the objects.

#### 2. SHAP Visualization

Once YOLOv8 generates predictions, SHAP is used to create explainability visualizations. These visualizations highlight the areas in the image that positively and negatively affect the model's decision for detecting each object.

#### 3. DRISE Saliency Maps

Additionally, DRISE is used to create saliency maps. These maps show the most important areas of the image, class-wise, that contributed to the object detection decision.

#### Running the Process

- The input directory containing the original images is specified in `input_dir`.
- The output directory where processed images and visualizations are saved is specified in `output_dir`.

The pipeline loops through all images and Visualizations (bounding boxes, SHAP, and DRISE) are generated automatically for each image.


---
### Part 2: Large Multimodal Model Interaction

The second part of the pipeline involves interaction with a large multimodal model, such as Idefics2. This model is provided with the visual outputs (bounding boxes, SHAP visualizations, saliency maps) and a contextual text prompt, which allows deeper interpretation of the model’s outputs. We can then proceed to use the model by passing some details such as the image name, class_id and the question about these images. The model then provides a detailed explanation with an answer to your question. 

#### Key Functions:

- **`load_images_for_analysis(image_name, class_id, input_dir)`**:
  Loads the images generated from YOLO, SHAP, and DRISE for a specific object class.

- **`generate_prompt_for_image(image_name, class_id, question)`**:
  Generates a prompt for querying the model based on the images and a user-defined question.

- **`ask_model(image_name, class_id, question, input_dir)`**:
  Sends the prompt to the large multimodal model for analysis and displays the results.

---
#### Enhanced SHAP Pipeline
The updated SHAP pipeline now leverages:

- SLIC (Simple Linear Iterative Clustering) for precise superpixel segmentation.

- Visualization improvements including clearer blending and optional bounding box highlighting.

## Explanation of Results

- **Bounding Box Visualization**: Shows where the model detects the object of interest.
- **SHAP Visualization**: Highlights regions of the image that influenced the model's decision, both positively and negatively.
- **Saliency Map**: Emphasizes the parts of the image that were most critical for the model's detection using the DRISE technique.

These visual outputs, when queried using the large multimodal model, provide insights into how the object detection model works internally and why certain decisions are made.

### Benefits

- Provides transparent insights into AI decision-making.

- Enables non-experts to intuitively understand complex AI predictions.

- Combines visual explainability and natural language understanding for comprehensive insights.


## Conclusion
By integrating object detection, visual explanations, and multimodal interactions, this pipeline bridges the gap between complex AI models and non-technical users, making AI decisions transparent and understandable.
