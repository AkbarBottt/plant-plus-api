# =========================================================
# PREDIKSI GAMBAR DAUN MANGGA
# Tahap 10 – Memuat model dan memprediksi input
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input

# =========================================================
# Inisialisasi Model (harus sama seperti saat training)
# =========================================================
classes = ['Anthracnose','Bacterial Canker','Cutting Weevil','Die Back',
           'Gall Midge','Healthy','Powdery Mildew','Sooty Mould']

base_model = EfficientNetB7(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3),
    pooling='max'
)
base_model.trainable = False

model = Sequential([
    base_model,
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.45),
    Dense(len(classes), activation='softmax')
])

# Memuat weights
model.load_weights("my_model_weights-v1-final.h5")

# =========================================================
# Fungsi prediksi
# =========================================================
def predict_and_return(image_path):
    from tensorflow.keras.preprocessing import image
    import numpy as np

    classes = ['Anthracnose','Bacterial Canker','Cutting Weevil','Die Back',
               'Gall Midge','Healthy','Powdery Mildew','Sooty Mould']

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = classes[predicted_class_index]
    confidence = float(np.max(prediction))

    return {
        "disease": predicted_class_label,
        "confidence": confidence
    }
