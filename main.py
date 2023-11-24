import cv2
from ultralytics import YOLO
import easyocr
import csv

model = YOLO('best.pt')

video_path = "vid.mp4"
cap = cv2.VideoCapture(video_path)

reader = easyocr.Reader(['en'])

ocr_results = []

while cap.isOpened():
  
    success, frame = cap.read()

    if success:

        results = model(frame)
        bounding_boxes = [detection.boxes for detection in results]

        print("Number of bounding boxes:", len(bounding_boxes))  

        if bounding_boxes:
            for bbox in bounding_boxes:
                if bbox.xyxy.shape[0] > 0:  
                    x_min, y_min, x_max, y_max = bbox.xyxy[0]
                    cropped_object = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

                    ocr_result = reader.readtext(cropped_object)

                    extracted_text = [text for _, text, _ in ocr_result]
                    ocr_results.append(extracted_text)

        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()

csv_file = 'ocr_results.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Detected Text'])
    for result in ocr_results:
        writer.writerow(result)
