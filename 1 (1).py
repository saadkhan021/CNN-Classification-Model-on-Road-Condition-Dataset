import zipfile
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. Extract ZIP

zip_path = r"C:\Users\Admins\Desktop\DM & CV project\RoadAlert Dataset.zip"
extract_path = "RoadAlert_Dataset"

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

print(" Dataset extracted")


# 2. Check Folder Structure

print("Classes found:", os.listdir(extract_path))


# 3. Data Generator (CORRECT)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    extract_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',   # VERY IMPORTANT
    subset='training'
)

val_gen = datagen.flow_from_directory(
    extract_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',   # VERY IMPORTANT
    subset='validation'
)

print("Class indices:", train_gen.class_indices)
print("Train label shape:", train_gen[0][1].shape)
