from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO # To handle byte streams
import torch
from transformers import ViTForImageClassification, ViTImageProcessor


app = FastAPI()


# Loads the model and processor
model_id = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_id)
processor = ViTImageProcessor.from_pretrained(model_id)


# Moves model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Funtion to normalize the predicted label
def normalize_label(label):
    label = label.lower().strip()
    if ',' in label:
        label = label.split(',')[0].strip()

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

    normalized_label = mappings.get(label, label)

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

    group = None
    for group_name, items in groups.items():
        if any(item in normalized_label for item in items):
            group = group_name.split()[1]
            break

    return normalized_label, group



# Defines the endpoint to process image uploads
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = logits.argmax(-1).item()
        confidence = probabilities[0][predicted_class_idx].item() * 100

        labels = model.config.id2label
        original_label = labels[predicted_class_idx]
        normalized_label, group = normalize_label(original_label)

        result = {
            "device": normalized_label.title(),
            "confidence": f"{confidence:.2f}%",
            "group": group
        }
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
