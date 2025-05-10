import cv2
import mediapipe as mp
import pygame
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time
import os
from tkinter import *
from tkinter import ttk, messagebox
from pydub import AudioSegment
from pydub.utils import which
from PIL import Image, ImageTk

class MusicApp:
    def __init__(self, root):

        self.root = root
        self.root.title("Music with Hands")
        self.root.geometry("400x300")
        self.root.attributes("-fullscreen", True)

        self.root.configure(bg="#222")
        
        self.bg_image = Image.open("image/handmusic.jpg")
        self.bg_image = self.bg_image.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        self.bg_label = Label(self.root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # notes
        self.notes = {
            1: "do",
            2: "re",
            3: "mi",
            4: "fa",
            5: "sol",
            6: "la",
            7: "si"
        }
        self.prev_finger_number = {0: None, 1: None}  # finger num for per hand
        self.last_sing_note = {0: None, 1: None} 
        self.last_sing_time = {} # when latest note singed
        self.sing_range = 0.2 # duration for singing same note again (sec)

        self.colors = {
            "do": "#ff4c4c",
            "re": "#ff914d",
            "mi": "#fff94d",
            "fa": "#5dff4d",
            "sol": "#4dcfff",
            "la": "#ae4dff",
            "si": "#ff4df1"
        }

        pygame.mixer.init()
        self.note_sounds = {}
        for index, file in self.notes.items():
            path_mp3 = f"notes/{file}.mp3"
            path_wav = f"notes/{file}.wav"
            if os.path.exists(path_mp3) and not os.path.exists(path_wav):
                sound = AudioSegment.from_mp3(path_mp3)
                sound.export(path_wav, format="wav")
            if os.path.exists(path_wav):
                self.note_sounds[file] = pygame.mixer.Sound(path_wav)
                self.last_sing_time[file] = 0

        ttk.Style().configure("TButton", padding=10, relief="flat", font=("Helvetica", 16),width = 20)
        self.btn_start = ttk.Button(root, text="Start", command=self.start)
        self.btn_start.place(relx=0.2, rely=0.2, anchor=CENTER)

        self.btn_exit = ttk.Button(root, text="Exit", command=root.quit)
        self.btn_exit.place(relx=0.9, rely=0.9, anchor=CENTER)


        self.record_state = False
        self.record = []
        self.thread = None
        self.cap = None

    def start(self):
        self.record_start()
        self.cam_flow()

    def cam_flow(self):
        self.cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    continue

                image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                upt_finger_numbers = {0: None, 1: None}

                if results.multi_hand_landmarks:
                    for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        finger_numbers = self.fingr_num(hand_landmarks)
                        upt_finger_numbers[index] = finger_numbers

                        if 1 <= finger_numbers <= 7:
                            note = self.notes.get(finger_numbers)
                            if note and note in self.note_sounds:
                                now = time.time()
                                if self.prev_finger_number[index] != finger_numbers or (now - self.last_sing_time.get(note, 0) > self.sing_range and self.last_sing_note[index] != note):
                                    self.note_sounds[note].play()
                                    self.last_sing_note[index] = note
                                    self.last_sing_time[note] = now
                                    h, w, _ = image.shape
                                    x, y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                                    cv2.circle(image, (x, y), 30, self.hex_to_bgr(self.colors[note]), -1)
                                    cv2.putText(image, note.upper(), (x-20, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                        else:
                            self.last_sing_note[index] = None 

            
                if not results.multi_hand_landmarks or all(count == 0 for count in upt_finger_numbers.values() if count is not None):
                    for note_name in self.note_sounds:
                        self.note_sounds[note_name].stop()
                        self.last_sing_note = {0: None, 1: None}

                self.prev_finger_number = upt_finger_numbers

                cv2.imshow("Music Cam", image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

            self.stop_record()
            self.cap.release()
            cv2.destroyAllWindows()

    def fingr_num(self, hand_landmarks):
        fingers = []
       
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
        for tip_id in [8, 12, 16, 20]:
            fingers.append(1 if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y else 0)
        return sum(fingers)

    def hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

    def record_start(self):
        self.record = []
        self.record_state = True
        def mikrofon_kayit():
            with sd.InputStream(callback=lambda indata, *_: self.record.append(indata.copy())):
                while self.record_state:
                    sd.sleep(100)
        self.thread = threading.Thread(target=mikrofon_kayit)
        self.thread.start()

    def stop_record(self):
        self.record_state = False
        if self.record:
            os.makedirs("records", exist_ok=True)
            file_name = f"records/record_{int(time.time())}.wav"
            sf.write(file_name, np.concatenate(self.record), 44100)
            print("Record saved:", file_name)

if __name__ == "__main__":
    root = Tk()
    
    app = MusicApp(root)
    root.mainloop()