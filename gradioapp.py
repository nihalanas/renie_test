import gradio as gr
import requests
from io import BytesIO

API_URL = "http://127.0.0.1:8000/predict/"

def predict_object(image):
    try:

        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Prepares files for HTTP POST request
        files = {"file": ("image.jpg", buffer, "image/jpeg")}
        response = requests.post(API_URL, files=files)

        # Handles the response
        if response.status_code == 200:
            result = response.json()
            return (f"Device: {result['device']}\n"
                    f"Confidence: {result['confidence']}\n"
                    f"Group: {result['group']}")
        else:
            return f"Error: {response.json().get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error: {str(e)}"


# Defines Gradio interface
interface = gr.Interface(
    fn=predict_object,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Textbox(label="Detection Result"),
    title="E-Waste Object Detection and Classification",
    description="Upload an image of e-waste to identify the object and its e-waste category.",
)

interface.launch(share = True)

