# Barcode-Based Information Extraction with OCR

This Python script performs barcode detection within an image using YOLOv5 and then extracts key information around the detected barcode using Paddle OCR.

# Overview

The script aims to locate barcodes within an image and subsequently extract text information from the identified barcode region. It integrates the YOLOv5 model for barcode detection and utilizes Paddle OCR to perform Optical Character Recognition (OCR) on the detected barcode area.



# Barcode Detection:

Utilizes YOLOv5 to detect the location of barcodes within an image.

# Information Extraction:

Extracts the region around the detected barcode.

Performs OCR on this region using Paddle OCR to retrieve text information.
