from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

IMG_SIZE = 128

train = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "dataset_body",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32
)

model = models.Sequential([
    layers.Conv2D(32,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(4,activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(train, epochs=5)

model.save("models/body_type_model.h5")