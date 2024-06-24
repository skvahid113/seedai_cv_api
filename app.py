import base64
from fastapi import FastAPI, HTTPException, Query, Header, File, UploadFile, HTTPException
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
from fastapi.responses import JSONResponse
import requests
import json
import os
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import vertexai
import base64
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from google.cloud import aiplatform
from fastapi import FastAPI, HTTPException, Query
from typing import List
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
import base64
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from google.cloud import aiplatform

app = FastAPI()

project_id = "860103948994"
endpoint_id_od = "2151469377650688000"
location = "us-central1"
api_endpoint = "us-central1-aiplatform.googleapis.com"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Define API key constants
API_KEY = "AIzaSyD7DbIF_nStfpj1vjIVpX7q-eBrM-YcRbI"

endpoint_id = "2416733154921414656"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print("file.....",file.read())
        # Read the file content
        content = await file.read()
        print("content....",content)
        # Encode the content as base64
        encoded_content = base64.b64encode(content).decode("utf-8")
        # Initialize the AI Platform client
        client_options = {"api_endpoint": api_endpoint}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        # Create the instance and parameters for the prediction request
        instance = {"content": encoded_content}
        instances = [instance]
        parameters = {"confidenceThreshold": 0.0, "maxPredictions": 5}  # Adjust confidenceThreshold if necessary

        # Define the endpoint path
        endpoint = client.endpoint_path(
            project=project_id, location=location, endpoint=endpoint_id
        )

        print("endpoint.....",endpoint)
        # Make the prediction request
        response = client.predict(
            endpoint=endpoint, instances=instances, parameters=parameters
        )

        # Log the entire prediction response for debugging
        logger.info("Prediction response: %s", response)
        print("response.....",response)

        # Extract predictions from the response
        predictions = response.predictions
        results = []
        for prediction in predictions:
            display_names = prediction.get("displayNames", [])
            confidences = prediction.get("confidences", [])
    
    # Iterate through displayNames and confidences
            for i in range(min(len(display_names), len(confidences))):
                label = display_names[i]
                score = confidences[i]
                results.append({"label": label, "score": score})

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Initialize Vertex AI
vertexai.init(project="seedai-421406", location="us-central1")

# Define generative model initialization
model = GenerativeModel("gemini-1.5-flash-001")

generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

@app.post("/generate")
async def generate_content(text: str = Query(..., description="Text for content generation")):
    prompt = "you are compost predictor. Based on the objects that are detected, suggest the detected objects for compost"
    print("request",text)
    try:
        responses = model.generate_content(
            [prompt +"/n /n"+ text],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

        generated_responses = []
        for response in responses:
            generated_responses.append(response.text)

        return {"generated_text": generated_responses}  # Assuming you want to return the first generated response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Generate descriptions for detected objects
def generate_descriptions(detected_objects):
    descriptions = []
    for obj in detected_objects:
        description = f"The image contains {obj['label']} with a confidence score of {obj['score']:.2f}."
        descriptions.append(description)
    return descriptions

# Function to crop and display detected objects
def crop_detected_objects(image_content, detected_objects):
    image = Image.open(BytesIO(image_content))
    width, height = image.size
    cropped_images = []
    
    for obj in detected_objects:
        vertices = obj['vertices']
        left = vertices[0][0] * width
        top = vertices[0][1] * height
        right = vertices[2][0] * width
        bottom = vertices[2][1] * height
        
        # Crop the object
        cropped_image = image.crop((left, top, right, bottom))
        cropped_images.append(cropped_image)
        
    return cropped_images

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_data = await file.read()

        # Encode image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Call Vision API to Detect Objects
        vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"
        payload = {
            "requests": [
                {
                    "image": {
                        "content": image_base64
                    },
                    "features": [
                        {
                            "type": "OBJECT_LOCALIZATION",
                            "maxResults": 10
                        }
                    ]
                }
            ]
        }

        response = requests.post(vision_api_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
        response_data = response.json()

        # Check for errors in the response
        if 'error' in response_data:
            raise HTTPException(status_code=400, detail=response_data['error']['message'])
        
        # Process the response
        detected_objects = []
        if 'responses' in response_data and 'localizedObjectAnnotations' in response_data['responses'][0]:
            objects = response_data['responses'][0]['localizedObjectAnnotations']
            for object_ in objects:
                label = object_['name']
                score = object_['score']
                vertices = [(vertex['x'], vertex['y']) for vertex in object_['boundingPoly']['normalizedVertices']]
                detected_objects.append({'label': label, 'score': score, 'vertices': vertices})
            
            # Generate descriptions for detected objects
            descriptions = generate_descriptions(detected_objects)
            
            # Crop and display detected objects
            cropped_images = crop_detected_objects(image_data, detected_objects)
            
            # Convert cropped images to base64 for inclusion in response
            cropped_images_base64 = [base64.b64encode(img.tobytes()).decode('utf-8') for img in cropped_images]
            
            return JSONResponse(content={
                "detected_objects": detected_objects,
                "descriptions": descriptions,
                "cropped_images": cropped_images_base64
            })

        return JSONResponse(content={"message": "No objects detected."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_objects/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the file content
        content = await file.read()

        # Encode the content as base64
        encoded_content = base64.b64encode(content).decode("utf-8")

        # Initialize the AI Platform client
        client_options = {"api_endpoint": api_endpoint}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        # Create the instance and parameters for the prediction request
        instance = {"content": encoded_content}
        instances = [instance]
        parameters = {"confidenceThreshold": 0.0, "maxPredictions": 5}  # Adjust confidenceThreshold if necessary

        # Define the endpoint path
        endpoint = client.endpoint_path(
            project=project_id, location=location, endpoint=endpoint_id_od
        )

        # Make the prediction request
        response = client.predict(
            endpoint=endpoint_id_od, instances=instances, parameters=parameters
        )

        # Log the entire prediction response for debugging
        logger.info("Prediction response: %s", response)

        # Extract predictions from the response
        predictions = response.predictions
        results = []
        for prediction in predictions:
            display_names = prediction.get("displayNames", [])
            confidences = prediction.get("confidences", [])
    
    # Iterate through displayNames and confidences
            for i in range(min(len(display_names), len(confidences))):
                label = display_names[i]
                score = confidences[i]
                results.append({"label": label, "score": score})

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# Run the FastAPI app with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
