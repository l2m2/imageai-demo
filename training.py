from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="answer-sheet")
trainer.setTrainConfig(object_names_array=["t0", "t1", "t2", "t10"], batch_size=4, num_experiments=200, train_from_pretrained_model="yolov3.pt")
# In the above,when training for detecting multiple objects,
#set object_names_array=["object1", "object2", "object3",..."objectz"]
trainer.trainModel()