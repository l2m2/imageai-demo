from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="answer-sheet2")
trainer.setTrainConfig(object_names_array=[
"dog",
"person",
"cat",
"tv",
"car",
"meatballs",
"marinara sauce",
"tomato soup",
"chicken noodle soup",
"french onion soup",
"chicken breast",
"ribs",
"pulled pork",
"hamburger",
"cavity",
"a",
"b",
"c",
"d"
], batch_size=4, num_experiments=20, train_from_pretrained_model="yolov3.pt")
# In the above,when training for detecting multiple objects,
#set object_names_array=["object1", "object2", "object3",..."objectz"]
trainer.trainModel()