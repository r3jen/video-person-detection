# Video Person Detection App

A Streamlit application that detects people in uploaded videos using MobileNet SSD model.

## Features

- Upload video files (MP4, AVI, MOV)
- Real-time person detection using MobileNet SSD
- Shows the original video alongside detection results
- Captures and displays the first frame where a person is detected
- Shows timestamp of first person detection
- User-friendly interface with Streamlit

## Requirements

- Python 3.7+
- OpenCV
- Streamlit
- NumPy

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Upload a video file using the sidebar
3. Click "Proses Video" to start detection
4. View results in the right column

## Model Files

Make sure you have the following model files in the `model` folder:
- `deploy.prototxt`
- `mobilenet_iter_73000.caffemodel`

## License

This project is open source and available under the MIT License. 