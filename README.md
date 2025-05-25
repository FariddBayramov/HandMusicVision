# 🎵 Hand Gesture Music Player

A real-time hand gesture recognition system that plays musical notes based on hand signs, using **MediaPipe**, **OpenCV**, and a trained **classification model**.

---

## 📸 How It Works

- Uses your webcam to track your hand.
- Detects specific hand gestures using a trained machine learning model.
- Maps each gesture to a corresponding musical note (A–G).
- Press **`P`** to play the note associated with the currently detected gesture.

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

5. Hold up a hand gesture and press **`P`** to play the corresponding note.
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
