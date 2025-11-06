# smart_waste_detector1
detects waste based on metal, paper and plastic
Overview

This project provides a real-time live waste detection system built using OpenCV and NumPy, without relying on deep learning models.
It uses image processing, color-space analysis, and texture-based heuristics to classify visible objects into one of three categories:

 Plastic

 Paper

 Metal

Additionally, the program uses text-to-speech (TTS) to audibly announce the detected waste type for user convenience.

Key Features

 Real-time detection from webcam feed
 Object segmentation using thresholding and contour analysis
 Feature extraction: color, brightness, specular reflection, texture
 Heuristic-based classification (Plastic, Paper, Metal)
 On-screen debugging of detection and feature metrics
 Optional text-to-speech alerts when waste type changes
 ROI-based detection for better accuracy
 Save ROI captures (s key) for dataset creation

 Techniques Used
1. Image Segmentation and Contour Extraction

Converts frames to grayscale and applies Gaussian Blur.

Uses Otsu’s thresholding to create a binary mask separating object and background.

Applies morphological operations (CLOSE, OPEN) to remove noise.

Extracts largest contour as the primary object region.

2. Feature Extraction

For the detected object region, the following features are computed:

Mean HSV Values (H, S, V): measures color tone, saturation, and brightness.

Specular Ratio: fraction of bright white pixels, useful for detecting shiny metallic surfaces.

Edge Ratio: proportion of Canny edges inside object mask; metals and papers have different edge densities.

Texture Standard Deviation: estimates surface roughness or smoothness.

3. Heuristic Classification

A rule-based scoring mechanism computes likelihoods for each waste type:

Plastic: high saturation (vibrant colors).

Paper: bright with low saturation.

Metal: strong specular reflection and edge density.

The highest-scoring label is selected unless confidence is too low → then labeled as “Unknown”.

4. Feedback and Visualization

Detected category is displayed on video frame.

Real-time numerical values (HSV, edge ratio, etc.) shown for debugging.

Uses pyttsx3 for spoken feedback when a new category is detected.
