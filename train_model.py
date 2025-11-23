# train_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# --------------------------
# 1. Préparation du dataset
# --------------------------
train_dir = "dataset/train"
val_dir   = "dataset/val"

# Data augmentation (IMPORTANT POUR 22 IMAGES)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen   = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=8,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=8,
    class_mode='binary'
)

print("Classes détectées :", train_gen.class_indices)
# doit afficher {'pleine':0 , 'vide':1}

# --------------------------
# 2. TRANSFER LEARNING : MobileNetV2
# --------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # gèle le modèle

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --------------------------
# 3. Entraînement
# --------------------------
history = model.fit(train_gen, validation_data=val_gen, epochs=12)

# --------------------------
# 4. Sauvegarde du modèle
# --------------------------
model.save("poubelle_model.h5")
print("Modèle enregistré !")
