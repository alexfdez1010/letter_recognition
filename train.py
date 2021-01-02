import numpy as np
from emnist import extract_training_samples,extract_test_samples

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

NUM_LETTERS = 26

def load_data() -> tuple:

    images_train, labels_train = extract_training_samples("letters")
    images_test, labels_test = extract_test_samples("letters")
    images = np.concatenate((images_train,images_test))
    labels = np.concatenate((labels_train,labels_test))
    images = np.expand_dims(images,axis = -1)
    labels = labels - 1
    return images,labels

def create_model():
    
    model = Sequential([
        
        Conv2D(128,(3,3),padding = "same",activation = "relu", input_shape = (28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(256,(3,3),padding = "same",activation = "relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(512, activation = "relu"),
        Dense(NUM_LETTERS, activation = "softmax")                
        ])
    
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])
    
    model.summary()
    
    return model

def train_model():
    
    images,labels = load_data()
    model = create_model()
    
    callback = EarlyStopping(patience = 2)
    
    model.fit(images,labels,batch_size = 32,epochs = 10, verbose = 1,callbacks = [callback],validation_split=0.15)
    
    model.save("letter_recognition.h5")
    
    return model 
