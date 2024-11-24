import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
# Activate v env
# ./fastml/Scripts/activate
# Load your pre-trained model
try:
    model = tf.keras.models.load_model('capstone-model2.h5')  # Adjust path if necessary
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Error loading model")

# Create the FastAPI app
app = FastAPI()

# Define the input data format for prediction
class PredictionResponse(BaseModel):
    prediction: list  # This will contain the predicted class probabilities or class index

# Function to preprocess the image for the model
def preprocess_image(image: Image.Image, target_size=(120, 120)):
    """
    Preprocess the image to match the model's input requirements.
    E.g., resizing and scaling pixel values.
    """
    try:
        # Resize image using the appropriate resampling filter (LANCZOS is a good choice)
        image = image.resize(target_size, resample=Image.Resampling.LANCZOS)
        
        # Convert to RGB if not already in that format
        image = image.convert("RGB")
        
        # Convert to numpy array and normalize (scale pixel values to [0, 1])
        image_array = np.array(image) / 1
        
        # If the model expects a batch dimension (e.g., for Keras models), add it
        image_array = np.expand_dims(image_array, axis=0)
        
        print(f"Preprocessed image shape: {image_array.shape}")
        return image_array
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Error in image preprocessing")

# Define the image prediction endpoint
@app.post("/predict-image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    try:
        # Debugging: Check file metadata and content
        print(f"Received file: {file.filename}, Content-Type: {file.content_type}")
        
        # Read the image file
        image_bytes = await file.read()
        print(f"Image bytes length: {len(image_bytes)}")  # Check the length to make sure it's not empty

        # Validate image bytes before processing
        if not image_bytes:
            raise HTTPException(status_code=400, detail="No image data found")
        
        # Convert to PIL Image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            print(f"Received image size: {image.size}")
        except Exception as e:
            print(f"Error reading image: {e}")
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Run the prediction
        prediction = model.predict(processed_image)
        print(f"Prediction result: {prediction}")
        
        # Get the predicted class probabilities (or class label if it's classification)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class index
        
        # Convert numpy.int64 to Python native int before returning it
        predicted_class = int(predicted_class)
        print(f"Predicted class: {predicted_class}")
        
        return PredictionResponse(prediction=[predicted_class])  # Return the predicted class index
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
