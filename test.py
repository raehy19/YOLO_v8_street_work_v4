from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Train the model
    train_results = model.train(data="data.yaml", epochs=10)

    # Validate the model
    val_results = model.val()  # no arguments needed, dataset and settings remembered

    # Predict with the model
    # predict_results = model()  # predict on an image

    # Export the model
    model.export()

