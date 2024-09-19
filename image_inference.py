from ultralytics import YOLO
import supervision as sv
import numpy as np

CONFIDENCE = 0.5  # Confidence threshold

def detect_and_annotate(image) -> any:
    """
    Detects birds and people in an image and returns the annotated image.
    
    Args:
        image_path (str): Path to the input image.

    Returns:
        annotated_image: The image with bounding boxes for birds and people.
    """

    # Convert the PIL image to a NumPy array
    image = np.array(image)

    # Load YOLO model
    model = YOLO("model/yolov8l.pt")

    # Box annotator to display bounding boxes
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # Search for bird and person class IDs
    bird_class_id = None
    person_class_id = None
    for class_id, name in model.model.names.items():
        if name.lower() == "bird":
            bird_class_id = class_id
        if name.lower() == "person":
            person_class_id = class_id

    # Run the YOLO model on the input image
    result = model(image, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)

    # Filter detections for birds and people with confidence > 0.5
    bird_people_detections = detections[
        ((detections.class_id == bird_class_id) | (detections.class_id == person_class_id)) 
        & (detections.confidence > CONFIDENCE)
    ]

    # Prepare labels for the detections
    labels = [
        f"{'heron' if model.model.names[class_id] == 'bird' else model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _ in bird_people_detections
    ]

    # Annotate the image with bounding boxes and labels
    annotated_image = box_annotator.annotate(
        scene=image, 
        detections=bird_people_detections, 
        labels=labels
    )

    return annotated_image
