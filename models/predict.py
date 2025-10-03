import tensorflow as tf
import numpy as np

# Load the .h5 model
model = tf.keras.models.load_model("F:/PROJECT/Trial000/models/allergy_detector.h5")

# Example input data (replace with your actual data)
input_data = np.random.rand(10, *model.input_shape[1:])  # 10 samples

# Get predictions
predictions = model.predict(input_data)

# Save predictions as CSV
np.savetxt("F:/PROJECT/Trial000/models/predictions.csv", predictions, delimiter=",")

print("Predictions saved successfully!")