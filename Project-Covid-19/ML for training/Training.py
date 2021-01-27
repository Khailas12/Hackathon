import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

print("Num of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

train_img = ImageDataGenerator(rescale=1.0/255)
test_img = ImageDataGenerator(rescale=1.0/255)

training_img = train_img.flow_from_directory(r"C:\Users\Admin\Documents\Project-covid-19\train", 
    target_size=(64, 64), batch_size=8, class_mode="binary")

testing_img = test_img.flow_from_directory(r"C:\Users\Admin\Documents\Project-covid-19\test", 
    target_size=(64, 64),batch_size=8, class_mode="binary")

print("Image Scanning Done")

def plotImages(images):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

sample_training, _ = next(training_img)
plotImages(sample_training[:5])

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

print("Training model")
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])
history = model.fit_generator(training_img, epochs=5, validation_data=testing_img)


acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epchos_range = range(5)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epchos_range, acc, label="Training Accuracy")
plt.plot(epchos_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("ACCURACY")

plt.subplot(1, 2, 2)
plt.plot(epchos_range, loss, label="Traning Loss")
plt.plot(epchos_range, val_loss, label="Validation Loss")
plt.legend(loc="upper left")
plt.title("LOSS")
plt.show()
plt.savefig("plot.png")

model.save("model.h5")
print("Finished")














