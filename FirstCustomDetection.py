from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("answer-sheet/models/yolov3_answer-sheet_mAP-0.65715_epoch-191.pt")
detector.setJsonPath("answer-sheet/json/answer-sheet_yolov3_detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="2.png", output_image_path="2-detected.png")
for detection in detections:
  print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])