# Required packages
# !pip install -q transformers gradio pillow

from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import gradioapp as gr


model_id = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_id)
processor = ViTImageProcessor.from_pretrained(model_id)


# Moves model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def normalize_label(label):
    """
    Normalize the predicted label to match our e-waste categories.
    Returns the normalized label, the group the device belongs to. and confidence rate.
    """
    label = label.lower().strip()


    # Cleans label if comma exists
    if ',' in label:
        label = label.split(',')[0].strip()


    # Defined mappings for device categories(Extra needed according to given instructions)
    mappings = {
        "television": "Smart TVs",
        "television system": "Smart TVs",
        "television receiver": "Smart TVs",
        "tv": "Smart TVs",
        "computer": "Desktop Computers",
        "personal computer": "Desktop Computers",
        "pc": "Desktop Computers",
        "notebook": "Laptops",
        "mobile phone": "Smartphones",
        "cell phone": "Smartphones",
        "cellphone": "Smartphones",
        "mouse": "Portable Keyboard and Mouse",
        "keyboard": "Portable Keyboard and Mouse",
        "headphone": "Gaming Headsets",
        "monitor": "Desktop Computers",
        "ipod": "Portable Gaming Consoles",
        "electric fan": "Fans",
        "remote": "Game Controllers",
        "remote control": "Game Controllers",
        "joystick": "Game Controllers",
        "hard disc": "External Hard Drives",
        "hard drive": "External Hard Drives",
        "usb": "USB Drives",
        "flash drive": "USB Drives",
        "thumb drive": "USB Drives",
        "camera": "Digital Cameras",
        "webcam": "Digital Cameras",
        "power supply": "Chargers",
        "power adapter": "Chargers",
        "power cord": "Cables",
        "electronic device": "Portable Gaming Consoles",
        "display": "Monitors",
        "screen": "Monitors"
    }


    # Map the labels
    normalized_label = mappings.get(label, label)


    # Group mapping for devices
    groups = {
        "group 1": ["Cables", "Chargers", "Adapters", "USB Drives", "Smart Pens"],
        "group 2": ["Smartphones", "Power Banks", "Smartwatches", "Fitness Trackers", "Smart Rings",
                    "Bluetooth Earbuds", "Bluetooth Speakers", "Portable Chargers", "Power Banks", "Earphones",
                    "Portable Keyboard and Mouse", "External Hard Drives", "Portable Projectors",
                    "Portable Gaming Consoles", "Gaming Headsets", "Game Controllers", "Pocket Calculators",
                    "Digital Cameras", "E-readers", "Lenses", "Wireless Routers", "Smart Light Bulbs"],
        "group 3": ["Laptops", "Tablets", "Gaming Consoles", "Digital SLR Cameras", "Soundbars", "Drones",
                    "Projectors", "Home Security Systems", "Virtual Reality Headsets"],
        "group 4": ["Smart TVs", "Microwaves", "Air Fryers", "Fans", "E-Bikes / Electric Scooters",
                    "Home Theater Systems", "Desktop Computers", "Kitchen Machines"]
    }


    # Group gets mapped
    group = None
    for group_name, items in groups.items():
        if any(item in normalized_label for item in items):
            group = group_name.split()[1]
            break

    return normalized_label, group

def predict_object(image):
    """
    Process the uploaded image and return the predicted object and its group.
    """
    try:

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Gets the predicted class index and probabilities
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = logits.argmax(-1).item()
        confidence = probabilities[0][predicted_class_idx].item() * 100

        # Gets and normalize the predicted label
        labels = model.config.id2label
        original_label = labels[predicted_class_idx]
        normalized_label, group = normalize_label(original_label)

        # Formats the output for display
        return (f"Device: {normalized_label.title()}\n"
                f"Confidence: {confidence:.2f}%\n"
                f"Group: {group}")

    except Exception as e:
        return f"Error: {str(e)}"


# Defines the Gradio interface for the app
interface = gr.Interface(
    fn=predict_object,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Textbox(label="Detection Result"),
    title="E-Waste Object Detection and Classification",
    description="Upload an image of e-waste to identify the object and its e-waste category.",
)


# Launches Gradio app with sharing enabled
interface.launch(share=True)
