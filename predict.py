import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tqdm import tqdm
import pandas as pd
from imutils import paths
from keras.models import model_from_json
import json
from keras.preprocessing import image
import numpy as np

image_path = list(sys.argv)[1]
model_path = list(sys.argv)[2]

CUDA_VISIBLE_DEVICES=""
# load json and create model
json_file = open(model_path + '\\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_path + "\\model.h5")
print("Loaded model from disk")

image_names = []
df = pd.DataFrame()

for imagePath in paths.list_images(image_path):
	image_names.append(imagePath.split('\\')[-1])
	
df['image_names'] = image_names
	

predict_image = []
for i in tqdm(range(int((df.shape[0])))):
    img = image.load_img(image_path+ "\\"+str(df['image_names'][i]),target_size=(300,300))
    img = image.img_to_array(img)
    img = img/255
    predict_image.append(img)
X = np.array(predict_image)

proba = loaded_model.predict(X)
y_classes = proba.argmax(axis=-1)

df['prediction'] = y_classes
df.to_csv('output.csv',index=False)
print ("prediction done")