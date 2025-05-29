# 🎵 Hand Gesture Music Player

A real-time hand gesture recognition system that plays musical notes based on hand signs, using **MediaPipe**, **OpenCV**, and a trained **classification model**.

---

## ✨ Features

* **Real-time Hand Tracking:** Utilizes MediaPipe to accurately detect and track your hand's landmarks from a webcam feed.
* **Gesture Recognition:** Employs a pre-trained **Random Forest Classifier** to identify specific hand gestures in real time.
* **Musical Note Playback:** Maps detected gestures to musical notes (A-G), played via Pygame. Just press **`P`**!
* **Melody Recording & Playback:** Record sequences of notes you play and save them. You can then play back your recorded melodies.
* **Interactive GUI:** A user-friendly Tkinter interface displays the live camera feed, recognized gestures, and provides controls for camera, recording, and data collection.

 ![image](https://github.com/user-attachments/assets/e4908f82-d0db-4e90-930b-3163642cc632)
 
* **Gesture Data Collection:** A built-in feature to collect new hand gesture data, allowing you to train your model with custom gestures.

  
## 📸 How It Works

- Uses your webcam to track your hand.
- Detects specific hand gestures using a trained machine learning model.
- Maps each gesture to a corresponding musical note (A–G).
- Press **`P`** to play the note associated with the currently detected gesture.
- The GUI manages all these processes, showing the live feed, recognized gestures, and offering controls for various functionalities including recording your musical creations.

---

## 🧠 Gestures & Notes Mapping

| Gesture        | Musical Note |
|----------------|--------------|
| peace_sign     | C            |
| fist           | D            |
| thumbs_up      | E            |
| open_palm      | F            |
| call_me        | G            |
| index_finger   | A            |
| rock_on        | B            |

---

## 🛠 Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
your_project/
├── models/
│   ├── gesture_classifier.pkl
│   └── gesture_labels.pkl
├── notes/
│   ├── A.wav
│   ├── B.wav
│   └── ... (C to G)
├── data_creator_main.py
├── gesture_music_player.py
├── model_demo.py
├── model.py
├── README.md
└── requirements.txt
```

---

## ▶️ Running the App

1. Make sure you have a webcam connected.
2. Place your trained model files (`gesture_classifier.pkl` and `gesture_labels.pkl`) in the `models/` folder.
3. Ensure your note sound files (`A.wav` to `G.wav`) are in the `notes/` folder.
4. Run the app:

```bash
python gesture_music_player.py
```

5. Interact with the App:

- The GUI will open, displaying your live camera feed.
Hold your hand in front of the camera. The "Gesture" label will show the recognized gesture.
Press the P key on your keyboard to play the musical note corresponding to the currently detected gesture.
Use the "Start/Stop Recording" button to record a sequence of played notes.
Click "Save Melody" to save your recorded sequence.
Use the "List Melodies" and "Play Selected" buttons to manage and listen to your saved musical compositions.
To close the application, simply close the GUI window.
6. Press **`ESC`** to exit the app.

---

## 📌 Notes

- The app expects **21 hand landmarks**, with **x and y** coordinates, for a total of **42 features**.
- Trained using MediaPipe hand tracking data collected through a custom script.

---

## 📷 Powered By

- [MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
- [Pygame](https://www.pygame.org/)
- [Scikit-learn](https://scikit-learn.org/)

---

Enjoy making music with your hands! ✋🎶
