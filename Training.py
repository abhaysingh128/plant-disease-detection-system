import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

  
training_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    'train',
    labels="inferred",
    labels_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(256,256),
    shuffle=True
)

validation_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    'train',
    labels="inferred",
    labels_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(256,256),
    shuffle=True
)

resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(256,256),
  layers.experimental.preprocessing.Rescaling(1./255),
])

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

train=train.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (32,128,128,3)
n_classes = 3

cnn_model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

cnn_model.build(input_shape=input_shape)

cnn_model.summary()

cnn_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(
    learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

 training_history=cnn_model.fit(x=training_dataset,validation_data=validation_dataset,epochs=10)

 #Training set Accuracy
train_loss,train_acc=cnn_model.evaluate(training_set)
print('Training accuracy:', train_acc)   

#Validation set Accuracy
val_loss,val_acc=cnn_model.evaluate(validation_dataset)
print('Validation accuracy:',val_acc)

cnn_model.save('trained_plant_disease_model.keras')

training_history.history #Return Dictionary of history

print(training_history.history.keys())

epochs = [i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'],color='blue',label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Visualization of Accuracy Result')
plt.legend()
plt.show()