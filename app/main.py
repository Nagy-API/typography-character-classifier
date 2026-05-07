from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import io
import os


app = FastAPI(title="TMNIST Character Classifier")


# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "tmnist_ann_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "label_classes.npy")
INDEX_PATH = os.path.join(BASE_DIR, "app", "index.html")


# =========================
# Load model + classes once
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Classes file not found: {CLASSES_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)


# Pillow compatibility for resize
if hasattr(Image, "Resampling"):
    RESAMPLE = Image.Resampling.LANCZOS
else:
    RESAMPLE = Image.LANCZOS


# =========================
# Helper: open image safely
# =========================
def open_image_from_bytes(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Handle transparency by pasting on white background
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background.convert("RGB")
    else:
        image = image.convert("RGB")

    return image


# =========================
# Main preprocessing
# =========================
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Strong preprocessing pipeline:
    1) open image
    2) grayscale
    3) autocontrast
    4) auto invert if background is bright
    5) threshold to binary
    6) crop around the character
    7) resize while keeping aspect ratio
    8) center in a 28x28 black canvas
    9) normalize and flatten
    """

    image = open_image_from_bytes(image_bytes)

    # Convert to grayscale
    image = image.convert("L")

    # Improve contrast
    image = ImageOps.autocontrast(image)

    # Convert to numpy for analysis
    arr = np.array(image).astype("float32")

    # If image is mostly bright => probably white background + dark character
    # invert it to black background + white character
    if arr.mean() > 127:
        image = ImageOps.invert(image)

    # Binary threshold
    # after inversion, character should be bright, background dark
    image = image.point(lambda p: 255 if p > 80 else 0)

    # Crop around non-zero pixels
    bbox = image.getbbox()
    if bbox is None:
        raise HTTPException(status_code=400, detail="No visible character found in image.")

    # Add a small padding around character before crop
    left, top, right, bottom = bbox
    pad = 4
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(image.width, right + pad)
    bottom = min(image.height, bottom + pad)

    image = image.crop((left, top, right, bottom))

    # Resize so the longest side becomes 20 pixels
    w, h = image.size
    if w == 0 or h == 0:
        raise HTTPException(status_code=400, detail="Invalid cropped character.")

    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    image = image.resize((new_w, new_h), RESAMPLE)

    # Create 28x28 black canvas and center character
    canvas = Image.new("L", (28, 28), color=0)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas.paste(image, (x_offset, y_offset))

    # Convert to array
    final_array = np.array(canvas).astype("float32") / 255.0

    # Flatten to (1, 784)
    final_array = final_array.reshape(1, 784)

    return final_array


# =========================
# Routes
# =========================
@app.get("/")
def home():
    if not os.path.exists(INDEX_PATH):
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(INDEX_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    processed_image = preprocess_image(image_bytes)

    prediction_probs = model.predict(processed_image, verbose=0)[0]

    predicted_index = int(np.argmax(prediction_probs))
    predicted_label = str(classes[predicted_index])
    confidence = float(prediction_probs[predicted_index])

    # Optional debug/helpful info: top 3 predictions
    top_indices = np.argsort(prediction_probs)[::-1][:3]
    top_predictions = [
        {
            "label": str(classes[i]),
            "confidence": float(prediction_probs[i])
        }
        for i in top_indices
    ]

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "top_predictions": top_predictions
    }