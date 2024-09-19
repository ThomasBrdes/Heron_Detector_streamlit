import cv2
import argparse

from ultralytics import YOLO
import supervision as sv


CONFIDENCE = 0.5

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("model/yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # Search for bird and person class id
    bird_class_id = None
    person_class_id = None
    for class_id, name in model.model.names.items():
        if name.lower() == "bird":
            bird_class_id = class_id
            break
    for class_id, name in model.model.names.items():
        if name.lower() == "person":
            person_class_id = class_id
            break
    
    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)

        # Filter detections for birds and people with a certain confidence
        bird_people_detections = detections[
            ((detections.class_id == bird_class_id) | (detections.class_id == person_class_id)) 
            & (detections.confidence > CONFIDENCE)
        ]

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in bird_people_detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=bird_people_detections, 
            labels=labels
        )  
        
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()