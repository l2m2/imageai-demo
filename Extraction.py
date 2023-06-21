from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("answer-sheet/models/yolov3_answer-sheet_mAP-0.65715_epoch-191.pt")
detector.setJsonPath("answer-sheet/json/answer-sheet_yolov3_detection_config.json")
detector.loadModel()
detections, extracted_objects_array = detector.detectObjectsFromImage(input_image="3.png", output_image_path="3-detected.png", extract_detected_objects=True)

for detection, object_path in zip(detections, extracted_objects_array):
    print(object_path)
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    print("---------------")