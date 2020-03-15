# Day_night_image_classifier
This repo includes my work to classify whether the image is captured in day light or at night time

# Install the requirements with pip install -r requirement.txt

## The ipython notebook consists the code for training, validating and testing the model with accuracy and other metrics.

We have also saved the model at last which can be directly used by running the predict.py file to predict on your images directly.

Note:
* The model.json and model.h5 name should not be altered.
* The path should be absolute path with proper slashes
* The arguments sequence is "image folder path" and then the model json/h5 file path (both should be in the same folder)

eg :- python predict.py C:\\Users\\hp\\Downloads\\images_upright\\query\\night\\milestone\\Renamed\\ S:\\Day_night_image_classification

* The output of the script is a csv file which contains image names with the predicted labels with '0' for day and '1' for night.

# I have used a deep neural network with 1722 images with 70% for training and 30% for validation purpose. The class images disposal was at 4:6 ratio(day:night). The model was then tested on fresh 472 images taken from another source.

About the performance

* The model training accuracy is 99.25% (as the data is not enough for training a deep neural n/w)
* The model validation accuracy is 98.65%
* Below metrics report was found on the test data:
    Accuracy: : 75.42%
    F1 Score: : 73.41%
    Precision Score: : 74.67%
    Recall Score: : 83.84%
    
The data can be obtained from my drive if required. Please reach out to me at mkyadav0625@gmail.com
    
