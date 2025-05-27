import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
import threading
import csv
import os
import time

class HandDataCollector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hand Gesture Data Collector")
        self.root.geometry("400x250")

        self.label_var = tk.StringVar(value="peace_sign")
        self.collecting = False
        self.record_start_time = None
        self.data = []

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        self.cap = cv2.VideoCapture(0)

        self.setup_ui()

        self.running = True
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_timer()
        self.root.mainloop()

    def setup_ui(self):
        tk.Label(self.root, text="Gesture Label:").pack(pady=5)
        self.label_entry = ttk.Entry(self.root, textvariable=self.label_var, width=30)
        self.label_entry.pack(pady=5)

        self.btn_start = ttk.Button(self.root, text="Start Recording", command=self.start_recording)
        self.btn_start.pack(pady=5)

        self.btn_stop = ttk.Button(self.root, text="Stop Recording", command=self.stop_recording)
        self.btn_stop.pack(pady=5)

        self.timer_label = tk.Label(self.root, text="Recording Time: 0s", font=("Helvetica", 12), fg="green")
        self.timer_label.pack(pady=10)

    def start_recording(self):
        self.collecting = True
        self.record_start_time = time.time()
        print(f"[INFO] Recording started with label: {self.label_var.get()}")

    def stop_recording(self):
        self.collecting = False
        print(f"[INFO] Recording stopped. {len(self.data)} samples collected.")

        if self.data:
            os.makedirs("gesture_data", exist_ok=True)
            file_path = os.path.join("gesture_data", "hand_gestures.csv")
            file_exists = os.path.isfile(file_path)

            with open(file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label"]
                    writer.writerow(header)
                writer.writerows(self.data)

            print(f"[INFO] Saved to {file_path}")
            self.data = []

        self.record_start_time = None
        self.timer_label.config(text="Recording Time: 0s")

    def update_timer(self):
        if self.collecting and self.record_start_time:
            elapsed = int(time.time() - self.record_start_time)
            self.timer_label.config(text=f"Recording Time: {elapsed}s")
        self.root.after(500, self.update_timer)  # refresh every 0.5s

    def video_loop(self):
        while self.running and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                    if self.collecting:
                        coords_x = [lm.x for lm in hand_landmarks.landmark]
                        coords_y = [lm.y for lm in hand_landmarks.landmark]
                        row = coords_x + coords_y + [self.label_var.get()]
                        self.data.append(row)

            cv2.imshow("Live Feed - Press ESC to exit", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                self.on_close()

        self.cap.release()
        cv2.destroyAllWindows()

    def on_close(self):
        self.running = False
        self.collecting = False
        self.thread.join()
        self.root.destroy()

if __name__ == "__main__":
    HandDataCollector()
