# vitaguard.py
import speech_recognition as sr
import serial
import time
import joblib
import argparse

# Choose model: 'decision', 'nb', or 'nn'
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["decision", "nb", "nn"], default="decision")
parser.add_argument("--port", default=None, help="Serial port for Arduino (e.g., COM3 or /dev/ttyUSB0)")
parser.add_argument("--baud", type=int, default=9600)
args = parser.parse_args()

# Load model
if args.model == "decision":
    model = joblib.load("decision_tree.pkl")
elif args.model == "nb":
    model = joblib.load("naive_bayes.pkl")
else:
    # neural net path: we will use text-keyword matching for simplicity in real-time
    # If you want audio->mfcc->nn prediction, integrate librosa and preproc
    model = None

# Configure serial (optional)
arduino = None
if args.port:
    try:
        arduino = serial.Serial(args.port, args.baud, timeout=1)
        time.sleep(2)  # allow connection
        print("Connected to Arduino on", args.port)
    except Exception as e:
        print("Could not open serial port:", e)
        arduino = None

r = sr.Recognizer()
mic = sr.Microphone()

KEYWORDS = ["help", "sos", "emergency"]

print("Listening... say 'stop' to exit.")

with mic as source:
    r.adjust_for_ambient_noise(source, duration=1)

try:
    while True:
        with mic as source:
            print("Say something...")
            audio = r.listen(source, timeout=5, phrase_time_limit=4)
        try:
            text = r.recognize_google(audio).lower()
            print("You said:", text)
            # simple keyword check:
            matched = [k for k in KEYWORDS if k in text]
            if matched:
                print("Keyword detected:", matched)
                if arduino:
                    arduino.write(b"alert\n")
                    print("Sent alert to Arduino")
                else:
                    print("No Arduino connected — should trigger alert here")
            elif "stop" in text or "exit" in text or "quit" in text:
                print("Stopping by voice command.")
                if arduino:
                    arduino.write(b"stop\n")
                break
            else:
                print("No keyword in phrase.")
        except sr.WaitTimeoutError:
            print("No speech detected (timeout).")
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print("Speech recognition service error:", e)
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    if arduino:
        arduino.close()
    print("Exiting.")
