from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
from keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D,MaxPooling2D, BatchNormalization, Flatten,GlobalAveragePooling2D
import numpy as np
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5

model = Sequential()
model.add(Conv2D( 32,3,3,input_shape=(64,64,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu',kernel_initializer='random_uniform'))
model.add(Dense(1, activation='softmax')) 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
saved_model='static/Missing_2.h5'
model = load_model_from_hdf5(saved_model)


COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))
    img_arr = cv2.resize(img_arr, (64,64))
    img_arr = img_arr.reshape(1, 64,64,3)
    prediction = model.predict(img_arr)
    x = round(prediction[0,0], 2)
    preds = np.array([x])
    COUNT += 1
    return render_template('prediction.html', data=preds)
@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))
if __name__ == '__main__':
    app.run(debug=True)



