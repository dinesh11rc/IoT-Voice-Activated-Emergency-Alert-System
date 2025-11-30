# IoT-Voice-Activated-Emergency-Alert-System
VitaGuard is an IoT-based voice-activated emergency alert system that detects keywords like help, SOS, and emergency using machine-learning models. It processes audio in real time, classifies alerts, and triggers Arduino-based hardware responses such as buzzer or LED signals to ensure quick safety notifications.


Quick Setup & Run Instructions

Create virtualenv and install:

python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install numpy pandas scikit-learn joblib speechrecognition pyserial
# For neural network:
pip install tensorflow
# For advanced audio features (optional):
# pip install librosa sounddevice


Generate dataset:

python create_sos_dataset.py


Train models (pick one or all):

python decision_tree.py
python naive_bayes.py
python neural_network.py   # optional


Upload Arduino sketch via Arduino IDE to your board. Connect buzzer to pin 9 and LED to pin 13 (GND to GND, + as needed).

Run VitaGuard and (optionally) pass serial port:

python vitaguard.py --model decision --port COM3
# or on Linux
python vitaguard.py --port /dev/ttyUSB0


Speak "help", "sos" or "emergency" — Arduino will receive alert and activate buzzer/LED. Say "stop" to disable.
