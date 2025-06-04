# age-detection-
#Age Group Detection using CNN and OpenCV

This project detects faces in real-time using OpenCV and classifies them into predefined age groups using a trained Convolutional Neural Network (CNN) model built with TensorFlow/Keras.

## Project Overview

- A CNN model is trained to classify face images into one of six age groups.
- Real-time video is processed to detect faces using OpenCV.
- Detected faces are passed through the model for age prediction and labeled live on screen.

## Age Categories

The model classifies people into the following age ranges:
- 0-12
- 13-20
- 21-35
- 36-50
- 51-60
- 60+

## Files

- `age detection grp project.py`: Python script that performs live age detection using webcam
- `age_model.h5`: Trained CNN model (must be present at the specified path)

## Requirements

Install required packages using pip:

```bash
pip install tensorflow opencv-python numpy
```

## How to Run

1. Place the trained model `age_model.h5` in the appropriate directory
2. Run the script:

```bash
python "age detection grp project.py"
```

3. A webcam window will open showing detected faces with predicted age groups
4. Press `q` to quit the application

##  How It Works

- Uses Haar Cascade to detect faces
- Resizes detected face to 64x64
- Normalizes and passes the image through the CNN model
- Displays prediction as a label on the frame

## Notes

- Update the path to `age_model.h5` in the script if necessary
- Ensure your webcam is working and accessible

## License

This project is licensed under the MIT License.
