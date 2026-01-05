# UCR NeuroSpeed 2025

Top 10 Finalist - UIT Car Racing 2025.
Autonomous racing car control system using Computer Vision and Control Theory.

## Technologies

* **Lane Segmentation:** UNet
* **Traffic Sign Detection:** YOLO
* **Control Algorithm:** PID Controller
* **State Management:** Finite State Machine (FSM)
* **Language:** Python 3.8+

## Structure

* `assets/models`: Pre-trained weights for UNet and YOLO.
* `modules`: Core logic for perception and control.
* `client_lib.so`: Simulator communication library.
* `main.py`: Entry point.

## Setup & Usage

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run
```bash
python main.py
```
## Credits
DanhLent.
