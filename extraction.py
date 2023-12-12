import torch
from PIL import Image
import numpy as np
import cv2

# Load YOLOv5 model
model = torch.hub.load('/Users/sivagar/Documents/projects/farrer_hos/projects/barcode/yolov5', 'custom', path='/Users/sivagar/Documents/projects/farrer_hos/projects/barcode/barcode.pt', source='local')
model.eval()

def detect_and_annotate(image):
    # Convert cv2 image to PIL image
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Perform prediction
    results = model(image_pil)

    # Get detection results
    pred = results.pred[0]
    pred_boxes = pred[:, :4].cpu().numpy()  # Get bounding boxes

    detected_barcodes = []

    x_min_new, y_min_new, x_max_new, y_max_new = None, None, None, None
    top_left_detected = False  # Initialize top left barcode detection to False
    bottom_right_detected = False  # Initialize bottom right barcode detection to False

    for box in pred_boxes:
        x_min, y_min, x_max, y_max = box
        detected_barcodes.append((x_min, y_min, x_max, y_max))

        print(f"Barcode detected at coordinates: x_min={x_min:.2f}, y_min={y_min:.2f}, x_max={x_max:.2f}, y_max={y_max:.2f}")

        # If barcode detected in top-left, draw a larger bounding box above it
        if y_min < image_pil.size[1] / 2 and x_min < image_pil.size[0] / 2:
            x_min_new = x_min 
            y_min_new = y_min - 2.8 * (y_max - y_min)
            x_max_new = x_max + 0.4 * (x_max - x_min)
            y_max_new = y_max + 0.3 * (y_max - y_min)

            enlarged_bbox = (x_min_new, y_min_new, x_max_new, y_max_new)
            detected_barcodes.append(enlarged_bbox)

            print(f"Enlarged bounding box coordinates (top left): x_min={x_min_new:.2f}, y_min={y_min_new:.2f}, x_max={x_max_new:.2f}, y_max={y_max_new:.2f}")
            
            top_left_detected = True  # Set top left detection to True

        # If barcode detected in bottom-right, set the flag to True
        if y_max > image_pil.size[1] / 2 and x_max > image_pil.size[0] / 2:
            bottom_right_detected = True

    return x_min_new, y_min_new, x_max_new, y_max_new, top_left_detected, bottom_right_detected



img =""
x_min_new, y_min_new, x_max_new, y_max_new, top_left_detected, bottom_right_detected = detect_and_annotate(img)



#ocr_results = my_dict[0]['ocr_result']

def extract_info_from_region(x_min_new, y_min_new, x_max_new, y_max_new, ocr_results):

    if x_min_new is None or y_min_new is None or x_max_new is None or y_max_new is None:
        print("Region coordinates are not defined. Cannot extract information.")
        return None
    
    # Initialize a list to store extracted texts in the specified region
    extracted_results = []

    #ocr_results = clasify_document[0]['ocr_result']

    # Iterate through the nested lists within the ocr_results
    for outer_list in ocr_results:
        for inner_list in outer_list:
            # Extract the coordinates from the inner list
            coordinates = inner_list[0]

            # Check if any point within the region falls within the specified coordinates
            region_in_bounds = any(
                x_min_new <= point[0] <= x_max_new and y_min_new <= point[1] <= y_max_new
                for point in coordinates
            )

            if region_in_bounds:
                # Extract the text from the inner list and add it to the list
                text = inner_list[1][0]
                extracted_results.append(text)

    # Define the mapping of keys to extracted results
    first_name = extracted_results[0].split(',')[1].strip()
    first_name = first_name.split(' (')[0]  # Remove text within parentheses
    
    # Extract the age from the "Gender" value
    gender_age = extracted_results[3].split('/')
    gender = gender_age[0].strip()
    age = int(gender_age[1].strip()) if len(gender_age) > 1 and gender_age[1].strip().isdigit() else None


    details = {
        "LAST_NAME": extracted_results[0].split(',')[0].strip(),
        "FIRST_NAME": first_name,
        "PATIENT_ID": extracted_results[1],
        "DATE_OF_BIRTH": extracted_results[2],
        "GENDER": gender,
        "AGE": age,
        "FA": extracted_results[8],
        "FP": extracted_results[4],
        "ADMISSION_TIME": extracted_results[6],
        "DR_NAME": extracted_results[7],
        "ADMISSION_DATE": extracted_results[5]
    }

    # Print the created dictionary
    print(details)
    return details



# Call the function to extract and print information
extract_info_from_region(x_min_new, y_min_new, x_max_new, y_max_new, ocr_results)


#################################################################################

import collections

# Define the keys you want to consider
keys_to_consider = ['LAST_NAME', 'FIRST_NAME', 'PATIENT_ID', 'DATE_OF_BIRTH', 'GENDER', 'AGE', 'FA', 'FP']

# Create a dictionary to store counts of each key-value pair
counts = collections.defaultdict(collections.Counter)

# Count occurrences of each key-value pair for the specified keys
for details in all_details:
    if details is not None:  # Check if the dictionary is not None
        for key, value in details.items():
            if key in keys_to_consider:
                counts[key][value] += 1

# Create a consolidated details dictionary with most frequent values for each key
consolidated_details = {}
for key, counter in counts.items():
    most_common_value, _ = counter.most_common(1)[0]
    consolidated_details[key] = most_common_value

# Print the consolidated details dictionary
print(consolidated_details)




#########################################################################################


#compare the lists

def compare_details(image_details, consolidated_details, keys_to_compare):
    if image_details is None:
        return {}  # Return an empty dictionary if input is None

    result_dict = {}

    # Compare each key-value pair in image_details with consolidated_details for specified keys
    for key in keys_to_compare:
        if key in image_details and key in consolidated_details:
            if image_details[key] == consolidated_details[key]:
                result_dict[key] = True
            else:
                result_dict[key] = False
        else:
            # Handle cases where the key is missing in either image_details or consolidated_details
            result_dict[key] = "Key Missing"  # Indicate that the key is missing in one of the dictionaries

    return result_dict


result_dict = compare_details(details, consolidated_details, keys_to_consider)

# Print the result dictionary
print(result_dict)


