# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:53:41 2020

@author: LJ
"""
import keras
import keras.backend as K
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, GlobalAveragePooling1D, Reshape
from keras.models import Sequential, model_from_json
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, RMSprop
from keras.callbacks import LambdaCallback

from dsdatagen import DSDataGen
#%%
model = Sequential()
model.add(Conv1D(100, 3, activation='relu', padding='same', input_shape=(10, 16)))
model.add(Conv1D(100, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(3))
model.add(Conv1D(160, 3, activation='relu', padding='same'))
model.add(Conv1D(160, 3, activation='relu', padding='same'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# print(model.summary())
#%%
model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

train = DSDataGen(r'E:\dataset\_shootrun\training_set.json')
traingen = train.generator(batch_size=8, is_shuffle=True)
# val = DSDataGen(r'E:\dataset\_shootrun\test_set.json')
# valgen = val.generator(batch_size=40, is_shuffle=True)

model.fit_generator(generator=traingen,
                    # validation_data=valgen,
                    steps_per_epoch=train.get_dataset_size() // 8,
                    # validation_steps=val.get_dataset_size() // 8,
                    epochs=50,
                    verbose=1)

#%%
import matplotlib.pyplot as plt

val = DSDataGen(r'E:\dataset\_shootrun\test_set.json')
valgen = val.generator(batch_size=80, is_shuffle=True)
testds, testlbl = next(valgen)
#%%
sns.set_style("whitegrid", {'axes.grid' : False})
plt.imshow(testds[53,:,:])
plt.show()
#%%
out = model.predict(testds)
#%%
import json
# serialize model to JSON
model_json = model.to_json()
with open("model-prototype.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model-weights-prototype.h5")
print("Saved model to disk")
#%%
# evaluate the model
scores = model.evaluate(testds, testlbl, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#%%
# load json and create model
json_file = open('prototype(03-12-2020)/model-prototype.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("prototype(03-12-2020)/model-weights-prototype.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(testds, testlbl, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
#%%
sns.set()
# Plot training & validation accuracy values
plt.plot(model.history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('acc-prototype.png')
plt.show()

plt.plot(model.history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('loss-prototype.png')
plt.show()
#%%
import numpy as np
from sklearn import metrics
def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
    
max_y_pred_test = np.argmax(out, axis=1)
max_y_test = np.argmax(testlbl, axis=1)

LABELS = ["Shoot",
          "Run"]

show_confusion_matrix(max_y_test, max_y_pred_test)
#%%
from sklearn.metrics import classification_report
print(classification_report(max_y_test, max_y_pred_test))