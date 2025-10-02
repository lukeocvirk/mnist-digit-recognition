from dataset import create_model
from sample import preprocess_images, predict_sample

def main() -> None:
    """Main function to train/test and sample model with."""

    # Run the model
    model = create_model()

    # Preprocess custom images
    samples = preprocess_images("assets")

    # Predict sample digits
    predict_sample(model, samples)

if __name__ == "__main__":
    main()
    