import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pygame
import os
import pandas as pd
import time
import threading
import csv

class HandMusicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HandMusicVision")
        self.root.geometry("1200x750")

        # --- Load Model and Label Encoder ---
        try:
            self.model = joblib.load("models/gesture_classifier.pkl")
            self.label_encoder = joblib.load("models/gesture_labels.pkl")
            print("Model and label encoder loaded.")
        except FileNotFoundError:
            messagebox.showerror("Error", "Model files not found. Please run 'model.py' script to train the model.")
            self.root.quit()
            return

        pygame.mixer.init()

        self.notes_folder = "notes"
        self.note_names = ["A", "B", "C", "D", "E", "F", "G"]
        self.note_sounds = {}
        for name in self.note_names:
            try:
                note_path = os.path.join(self.notes_folder, f"{name}.wav")
                if not os.path.exists(note_path):
                    raise FileNotFoundError(f"'{name}.wav' not found.")
                self.note_sounds[name] = pygame.mixer.Sound(note_path)
            except (pygame.error, FileNotFoundError) as e:
                messagebox.showerror("Error", f"Could not load '{name}.wav'. Please ensure it's in the 'notes' folder. Error: {e}")
                self.root.quit()
                return
        print("Note sounds loaded.")

        # --- Map Gesture Labels to Note Names ---
        self.gesture_to_note = {
            "peace_sign": "C",
            "fist": "D",
            "thumbs_up": "E",
            "open_palm": "F",
            "call_me": "G",
            "index_finger": "A",
            "rock_on": "B"
        }

        # --- MediaPipe Initialization ---
        self.mp_hands = mp.solutions.hands
        self.hands = None 
        self.mp_drawing = mp.solutions.drawing_utils

        # --- Camera Variables ---
        self.cap = None
        self.camera_running = False
        self.current_label = None 
        self.min_detection_confidence = 0.6 
        self.min_tracking_confidence = 0.6 

        # --- Recording Feature Variables ---
        self.is_recording = False
        self.recorded_notes = []
        self.recording_start_time = 0
        self.last_played_time = 0
        self.min_play_interval = 0.3
        self.recordings_folder = "recordings"
        os.makedirs(self.recordings_folder, exist_ok=True)

        # --- Data Collection Variables ---
        self.collecting_data = False
        self.data_collection_start_time = 0
        self.collected_samples = []
        self.data_output_file = os.path.join("gesture_data", "hand_gestures.csv")
        os.makedirs("gesture_data", exist_ok=True)

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # --- Key Bindings ---
        self.root.bind('<KeyPress-p>', self.play_note_on_press)
        # Optional: Key bindings for confidence adjustments
        # self.root.bind('<Up>', lambda event: self.adjust_confidence(0.05, 'detection'))
        # self.root.bind('<Down>', lambda event: self.adjust_confidence(-0.05, 'detection'))
        # self.root.bind('<Right>', lambda event: self.adjust_confidence(0.05, 'tracking'))
        # self.root.bind('<Left>', lambda event: self.adjust_confidence(-0.05, 'tracking'))

    # --- Helper Functions ---
    def extract_landmarks(self, hand_landmarks):
        """Extracts x and y coordinates from MediaPipe hand landmarks."""
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        return x_coords + y_coords

    # Uncomment this method to enable confidence adjustments via key bindings
    # def adjust_confidence(self, change, type):
    #     if type == 'detection':
    #         self.min_detection_confidence = max(0.1, min(1.0, self.min_detection_confidence + change))
    #     elif type == 'tracking':
    #         self.min_tracking_confidence = max(0.1, min(1.0, self.min_tracking_confidence + change))
        
    #     # Reinitialize MediaPipe Hands object when confidence threshold changes
    #     if self.camera_running:
    #         self.hands = self.mp_hands.Hands(
    #             static_image_mode=False, 
    #             max_num_hands=1, 
    #             min_detection_confidence=self.min_detection_confidence,
    #             min_tracking_confidence=self.min_tracking_confidence
    #         )
    #     print(f"Confidence: Det={self.min_detection_confidence:.2f}, Trk={self.min_tracking_confidence:.2f}")

    def setup_ui(self):
        # --- Left Panel: Camera and Gesture Recognition ---
        left_frame = ttk.Frame(self.root, width=600, height=680, relief=tk.GROOVE, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        left_frame.pack_propagate(False)

        ttk.Label(left_frame, text="Live Camera Feed", font=("Arial", 14, "bold")).pack(pady=5)
        self.camera_canvas = tk.Canvas(left_frame, width=640, height=480, bg="black")
        self.camera_canvas.pack(pady=10)

        self.gesture_label_display = ttk.Label(left_frame, text="Gesture: Waiting...", font=("Arial", 16, "bold"), foreground="blue")
        self.gesture_label_display.pack(pady=5)

        control_frame = ttk.Frame(left_frame)
        control_frame.pack(pady=10)

        self.btn_start_camera = ttk.Button(control_frame, text="Start Camera", command=self.start_camera)
        self.btn_start_camera.grid(row=0, column=0, padx=5, pady=5)

        self.btn_stop_camera = ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.btn_stop_camera.grid(row=0, column=1, padx=5, pady=5)

        self.btn_toggle_record = ttk.Button(control_frame, text="Start/Stop Recording", command=self.toggle_recording, state=tk.DISABLED)
        self.btn_toggle_record.grid(row=1, column=0, padx=5, pady=5)

        self.btn_save_recording = ttk.Button(control_frame, text="Save Melody", command=self.save_current_recording, state=tk.DISABLED)
        self.btn_save_recording.grid(row=1, column=1, padx=5, pady=5)

        self.recording_status_label = ttk.Label(left_frame, text="Recording Status: Off", font=("Arial", 12))
        self.recording_status_label.pack(pady=5)

        # --- Right Panel: Recordings and Data Collection ---
        right_frame = ttk.Frame(self.root, width=600, height=680, relief=tk.GROOVE, borderwidth=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        right_frame.pack_propagate(False)

        # Melodies Section
        ttk.Label(right_frame, text="Recorded Melodies", font=("Arial", 14, "bold")).pack(pady=5)
        self.melody_listbox = tk.Listbox(right_frame, height=10, width=50)
        self.melody_listbox.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.melody_listbox.bind("<<ListboxSelect>>", self.on_melody_select)

        melody_control_frame = ttk.Frame(right_frame)
        melody_control_frame.pack(pady=5)
        self.btn_list_melodies = ttk.Button(melody_control_frame, text="List Melodies", command=self.list_recordings)
        self.btn_list_melodies.grid(row=0, column=0, padx=5)
        self.btn_play_selected = ttk.Button(melody_control_frame, text="Play Selected", command=self.play_selected_melody, state=tk.DISABLED)
        self.btn_play_selected.grid(row=0, column=1, padx=5)

        ttk.Separator(right_frame, orient="horizontal").pack(fill="x", pady=15)

        # Data Collection Section
        ttk.Label(right_frame, text="Collect New Gesture Data", font=("Arial", 14, "bold")).pack(pady=5)
        ttk.Label(right_frame, text="Gesture Name:").pack(pady=5)
        self.data_label_var = tk.StringVar(value="new_gesture")
        self.data_label_entry = ttk.Entry(right_frame, textvariable=self.data_label_var, width=40)
        self.data_label_entry.pack(pady=5)

        self.btn_start_data_collection = ttk.Button(right_frame, text="Start Data Collection", command=self.start_data_collection, state=tk.DISABLED)
        self.btn_start_data_collection.pack(pady=5)
        self.btn_stop_data_collection = ttk.Button(right_frame, text="Stop Data Collection", command=self.stop_data_collection, state=tk.DISABLED)
        self.btn_stop_data_collection.pack(pady=5)

        self.data_collection_status_label = ttk.Label(right_frame, text="Data Collection Status: Off", font=("Arial", 12))
        self.data_collection_status_label.pack(pady=5)
        self.data_collection_timer_label = ttk.Label(right_frame, text="Time: 0s | Samples: 0", font=("Arial", 12))
        self.data_collection_timer_label.pack(pady=5)

    # --- Camera and Processing Functions ---
    def start_camera(self):
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not start camera. Another application might be using it.")
                return

            # Initialize MediaPipe Hands object with confidence thresholds
            self.hands = self.mp_hands.Hands(
                static_image_mode=False, 
                max_num_hands=1, 
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.camera_running = True
            self.btn_start_camera.config(state=tk.DISABLED)
            self.btn_stop_camera.config(state=tk.NORMAL)
            self.btn_toggle_record.config(state=tk.NORMAL)
            self.btn_start_data_collection.config(state=tk.NORMAL)
            self.camera_thread = threading.Thread(target=self.video_loop)
            self.camera_thread.daemon = True # Closes thread when main program exits
            self.camera_thread.start()
            print("Camera started.")

    def stop_camera(self):
        if self.camera_running:
            self.camera_running = False
            if self.cap:
                self.cap.release()
            self.hands = None
            self.btn_start_camera.config(state=tk.NORMAL)
            self.btn_stop_camera.config(state=tk.DISABLED)
            self.btn_toggle_record.config(state=tk.DISABLED)
            self.btn_save_recording.config(state=tk.DISABLED)
            self.btn_start_data_collection.config(state=tk.DISABLED)
            self.btn_stop_data_collection.config(state=tk.DISABLED)
            self.is_recording = False
            self.recording_status_label.config(text="Recording Status: Off", foreground="black")
            self.collecting_data = False
            self.data_collection_status_label.config(text="Data Collection Status: Off", foreground="black")
            self.data_collection_timer_label.config(text="Time: 0s | Samples: 0")
            self.camera_canvas.delete("all")
            self.gesture_label_display.config(text="Gesture: Camera Off")
            print("Camera stopped.")

    def video_loop(self):
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1) # Flip image horizontally

            # Convert BGR to RGB for MediaPipe processing
            # This step is crucial for MediaPipe to work correctly
            rgb_frame_for_mediapipe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame_for_mediapipe)

            self.current_label = "No Gesture Detected"
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the frame (still in BGR format)
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    features = self.extract_landmarks(hand_landmarks)
                    if len(features) == 42:
                        # --- Gesture Recognition ---
                        X_input_df = pd.DataFrame([features], columns=[f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)])
                        prediction = self.model.predict(X_input_df)[0]
                        self.current_label = self.label_encoder.inverse_transform([prediction])[0]

                        # --- Data Collection ---
                        if self.collecting_data:
                            coords_x = [lm.x for lm in hand_landmarks.landmark]
                            coords_y = [lm.y for lm in hand_landmarks.landmark]
                            row = coords_x + coords_y + [self.data_label_var.get()]
                            self.collected_samples.append(row)

            # --- GUI Updates ---
            # Convert BGR to RGBA for Tkinter Canvas
            # This conversion ensures the image colors are correct in the GUI
            self.root.after(0, self.update_gui_elements, frame.copy())
            
            # Small delay to reduce CPU usage and improve fluidity
            time.sleep(0.01)

        # Release camera resources when the loop finishes
        if self.cap:
            self.cap.release()
        self.camera_canvas.delete("all")
        self.gesture_label_display.config(text="Gesture: Camera Off")

    def update_gui_elements(self, frame):
        """Function to update camera feed and text labels in the GUI thread."""
        if not self.camera_running:
            return

        # Update camera feed on Canvas
        img_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # Convert BGR to RGBA for color correction
        img_rgba = cv2.resize(img_rgba, (640, 480)) # Resize to fit Canvas
        img_tk = tk.PhotoImage(data=cv2.imencode('.png', img_rgba)[1].tobytes())
        self.camera_canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
        self.camera_canvas.image = img_tk # Keep a reference

        # Update gesture label
        self.gesture_label_display.config(text=f"Gesture: {self.current_label}")

        # Update data collection counter
        if self.collecting_data:
            elapsed = int(time.time() - self.data_collection_start_time)
            self.data_collection_timer_label.config(text=f"Time: {elapsed}s | Samples: {len(self.collected_samples)}")

    def play_note_on_press(self, event=None):
        """Plays the note of the current gesture and records it when 'P' key is pressed."""
        if self.camera_running and self.current_label and self.current_label != "No Gesture Detected":
            note_to_play = self.gesture_to_note.get(self.current_label)
            if note_to_play and note_to_play in self.note_sounds and self.note_sounds[note_to_play]:
                if (time.time() - self.last_played_time > self.min_play_interval):
                    self.note_sounds[note_to_play].play()
                    self.last_played_time = time.time()
                    print(f"Played: {note_to_play} ({self.current_label})")
                    if self.is_recording:
                        self.recorded_notes.append({
                            'note': note_to_play,
                            'timestamp_relative': time.time() - self.recording_start_time
                        })
                # else:
                #     print("Playing too fast, or sound file not found.")
            # else:
            #     print(f"No note sound assigned for gesture '{self.current_label}'.")
        # else:
        #     print("Camera is not running or no gesture detected.")

    # --- Note Recording and Playback Functions ---
    def toggle_recording(self):
        if self.camera_running:
            self.is_recording = not self.is_recording
            if self.is_recording:
                self.recorded_notes = []
                self.recording_start_time = time.time()
                self.recording_status_label.config(text="Recording Status: RECORDING...", foreground="red")
                self.btn_save_recording.config(state=tk.DISABLED)
                print("Melody recording started.")
            else:
                self.recording_status_label.config(text="Recording Status: Paused", foreground="green")
                self.btn_save_recording.config(state=tk.NORMAL if self.recorded_notes else tk.DISABLED)
                print("Melody recording stopped.")
        else:
            messagebox.showwarning("Warning", "You must start the camera before starting a recording.")

    def save_current_recording(self):
        if self.recorded_notes:
            self._save_recording_to_file(self.recorded_notes)
            self.recorded_notes = []
            self.recording_status_label.config(text="Recording Status: Saved", foreground="green")
            self.btn_save_recording.config(state=tk.DISABLED)
        else:
            messagebox.showinfo("Info", "No notes recorded to save.")

    def _save_recording_to_file(self, notes_data):
        """Saves recorded notes to a file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.recordings_folder, f"melody_{timestamp}.txt")

        with open(filename, "w") as f:
            for note_info in notes_data:
                f.write(f"{note_info['note']},{note_info['timestamp_relative']}\n")
        messagebox.showinfo("Success", f"Melody saved: {filename}")
        self.list_recordings()

    def list_recordings(self):
        """Loads the list of recorded melodies into the GUI."""
        self.melody_listbox.delete(0, tk.END)
        recordings_list = []
        for f in os.listdir(self.recordings_folder):
            if f.startswith("melody_") and f.endswith(".txt"):
                recordings_list.append(f)

        if recordings_list:
            recordings_list.sort()
            for rec_file in recordings_list:
                self.melody_listbox.insert(tk.END, rec_file)
            self.btn_play_selected.config(state=tk.NORMAL)
        else:
            self.melody_listbox.insert(tk.END, "No melodies recorded yet.")
            self.btn_play_selected.config(state=tk.DISABLED)

    def on_melody_select(self, event=None):
        if self.melody_listbox.curselection():
            self.btn_play_selected.config(state=tk.NORMAL)
        else:
            self.btn_play_selected.config(state=tk.DISABLED)

    def play_selected_melody(self):
        selected_index = self.melody_listbox.curselection()
        if selected_index:
            filename = self.melody_listbox.get(selected_index[0])
            self.play_melody_from_file(filename)
        else:
            messagebox.showinfo("Info", "Please select a melody to play.")

    def play_melody_from_file(self, filename):
        """Plays notes from the specified melody file."""
        filepath = os.path.join(self.recordings_folder, filename)
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"Melody file not found: {filepath}")
            return

        notes_to_play = []
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    notes_to_play.append({'note': parts[0], 'timestamp_relative': float(parts[1])})

        if not notes_to_play:
            messagebox.showinfo("Info", "No notes found in the melody file.")
            return

        play_thread = threading.Thread(target=self._actual_play_melody, args=(notes_to_play, filename))
        play_thread.daemon = True
        play_thread.start()

    def _actual_play_melody(self, notes_to_play, filename_display):
        """Helper function that actually plays the melody (within a thread)."""
        print(f"Playing melody: {filename_display}")
        melody_start_time = time.time()

        for note_info in notes_to_play:
            note = note_info['note']
            relative_time = note_info['timestamp_relative']

            target_time = melody_start_time + relative_time

            while time.time() < target_time:
                time.sleep(0.001)

            if note in self.note_sounds and self.note_sounds[note]:
                self.note_sounds[note].play()
            # else:
            #     print(f"  -> Sound file for note '{note}' not found.")
        print("Melody playback finished.")

    # --- Data Collection Functions ---
    def start_data_collection(self):
        if self.camera_running:
            self.collecting_data = True
            self.collected_samples = []
            self.data_collection_start_time = time.time()
            self.btn_start_data_collection.config(state=tk.DISABLED)
            self.btn_stop_data_collection.config(state=tk.NORMAL)
            self.data_collection_status_label.config(text="Data Collection Status: STARTED...", foreground="red")
            print(f"Data collection started for: {self.data_label_var.get()}")
        else:
            messagebox.showwarning("Warning", "You must start the camera before starting data collection.")

    def stop_data_collection(self):
        if self.collecting_data:
            self.collecting_data = False
            self.btn_start_data_collection.config(state=tk.NORMAL)
            self.btn_stop_data_collection.config(state=tk.DISABLED)
            self.data_collection_status_label.config(text="Data Collection Status: Stopped", foreground="green")
            print(f"Data collection stopped. {len(self.collected_samples)} samples collected.")

            if self.collected_samples:
                file_exists = os.path.isfile(self.data_output_file)
                with open(self.data_output_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label"]
                        writer.writerow(header)
                    writer.writerows(self.collected_samples)
                messagebox.showinfo("Success", f"Data saved: {self.data_output_file}")
                self.collected_samples = []
            else:
                messagebox.showinfo("Info", "No samples collected.")
            self.data_collection_timer_label.config(text="Time: 0s | Samples: 0")

    def on_close(self):
        """Release resources when the application closes."""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        pygame.quit()
        self.root.destroy()

# --- Start the Application ---
if __name__ == "__main__":
    root = tk.Tk()
    app = HandMusicApp(root)
    root.mainloop()