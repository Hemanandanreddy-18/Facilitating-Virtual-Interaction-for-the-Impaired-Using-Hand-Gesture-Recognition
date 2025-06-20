import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tkinter as tk
from threading import Thread
import speech_recognition as sr

# Initialize hand gesture recognition variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("converted_keras_2/keras_model.h5", "converted_keras_2/labels.txt")
offset = 20
imgSize = 300
folder = "Data2"
labels = ["Can I have Bill, Please", "Do You Need Help", "Get up", "Have a Nice Day", "I Lost Everything",
          "Let's be Friends", "Thanks for Everything", "What is your Name", "Where is the Restroom"]

# Initialize speech recognition variables
recognizer = sr.Recognizer()
transcription = []


# Function for hand gesture recognition
def hand_gesture_recognition():
    global cap, detector, classifier, offset, imgSize, labels
    while True:
        success, img = cap.read()
        imgOutput = img.copy()

        hands, img = detector.findHands(img)

        if hands:
            predictions = []
            for hand in hands:
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                predictions.append(labels[index])

                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

            if len(predictions) == 2 and predictions[0] == predictions[1]:
                combined_result = predictions[0]
                cv2.putText(imgOutput, f"Prediction: {combined_result}", (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(imgOutput, "Mismatch, Try Again", (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Hand Gesture Recognition", imgOutput)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Function for speech-to-text recognition
def listen_to_speech():
    global transcription
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)
                text = recognizer.recognize_google(audio)
                transcription.append(text)
                print(f"You said: {text}")

            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                print("Service request failed.")
            except sr.WaitTimeoutError:
                pass


def show_transcription():
    full_text = "\n".join(transcription)
    text_output.delete(1.0, tk.END)
    text_output.insert(tk.END, full_text)


# Function to run the speech-to-text recognition in a thread
def start_speech_to_text():
    listen_thread = Thread(target=listen_to_speech)
    listen_thread.daemon = True
    listen_thread.start()


# Function to run the hand gesture recognition in a thread
def start_hand_gesture():
    gesture_thread = Thread(target=hand_gesture_recognition)
    gesture_thread.daemon = True
    gesture_thread.start()


# Create the main Tkinter window
root = tk.Tk()
root.title("Speech or Hand Gesture to Text")

# Buttons for options
speech_button = tk.Button(root, text="Speech to Text", command=start_speech_to_text, width=30)
speech_button.pack(pady=10)

gesture_button = tk.Button(root, text="Hand Gesture to Text", command=start_hand_gesture, width=30)
gesture_button.pack(pady=10)

# Textbox for displaying the transcription
text_output = tk.Text(root, height=10, width=50)
text_output.pack(pady=10)

show_button = tk.Button(root, text="Show Transcription", command=show_transcription)
show_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
