import numpy as np

from pathlib import Path
from PIL import Image
from tensorflow import keras

def preprocess_images(file_path: str) -> list:
    """Prepare all png images in the given folder for evaluation by
    the model. Note: All images should be white digits on a black
    background.
    
    :param file_path: Path to folder with images to preprocess.
    :returns: A list of tuples containing the numpy array of the image
    and the file name.
    """
    dir = Path(file_path)
    images = []

    for item in dir.iterdir():
        # Skip item if not a png file.
        if not item.is_file() or item.suffix.lower() != ".png":
            continue
        img = Image.open(item)

        # Convert image to greyscale and resize to 28x28.
        img = img.convert("L")
        img = img.resize((28, 28))

        # Convert image to a numpy array and store.
        img_array = np.array(img).astype("float32") / 255.0
        images.append((img_array, item.name))
    
    return images

def predict_sample(model, samples) -> None:
    """Given a list of samples, predict the right digits, and print.

    :param model: The digit recognition model.
    :param samples: The sample images.
    """
    for img_array, file_name in samples:
        logits = model.predict(img_array[np.newaxis, ...])
        digit = int(np.argmax(logits, axis=1)[0])
    
        print(f"{file_name}: Predicted digit = {digit}")
