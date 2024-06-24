import base64
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from google.cloud import aiplatform

app = FastAPI()

project_id = "860103948994"
endpoint_id = "4611771780334354432"
location = "us-central1"
api_endpoint = "us-central1-aiplatform.googleapis.com"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/detect_objects/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the file content
        content = await file.read()

        # Encode the content as base64
        encoded_content = base64.b64encode(content).decode("utf-8")
        print('encoded.......',encoded_content)
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

        # Make the prediction request
        response = client.predict(
            endpoint=endpoint, instances=instances, parameters=parameters
        )
        print('response.......',response)
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