import cv2
from deepface import DeepFace
import os
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import sqlite3

# Paths to the local directories of known faces and voted faces
db_path = 'faces'
voted_faces_path = 'voted_faces'
csv_file_path = 'faces/data.csv'  # Path to the CSV file containing person data
db_file_path = 'faces/voter_database.db'  # Path to the SQLite database

# List of parties for the user to vote for
parties = ["Party A", "Party B", "Party C", "Party D"]  # Adjust as needed

# Ensure the voted_faces directory exists
if not os.path.exists(voted_faces_path):
    os.makedirs(voted_faces_path)

# Create the SQLite database and tables if they don't exist
def create_database():
    conn = sqlite3.connect(db_file_path)
    cur = conn.cursor()
    
    # Create tables for storing person data and voted faces
    cur.execute('''
        CREATE TABLE IF NOT EXISTS person_data (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            gender TEXT
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS voted_faces (
            id INTEGER PRIMARY KEY,
            name TEXT,
            image BLOB,
            has_voted INTEGER DEFAULT 0,
            party TEXT DEFAULT NULL
        )
    ''')

    # Load data from CSV into person_data table if it hasn't been loaded already
    data = pd.read_csv(csv_file_path)
    cur.execute('SELECT COUNT(*) FROM person_data')
    if cur.fetchone()[0] == 0:
        data.to_sql('person_data', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()

# Create a tkinter application
class FaceRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Facial Recognition Voting System")
        self.geometry("1000x800")  # Adjust window size as needed
        
        # Create frames for different sections
        self.video_frame = tk.Frame(self)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)
        
        self.info_frame = tk.Frame(self)
        self.info_frame.grid(row=1, column=0, padx=10, pady=10, sticky="n")
        
        self.voting_frame = tk.Frame(self)
        self.voting_frame.grid(row=2, column=0, padx=10, pady=10, sticky="n")

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Create a label to display the video feed
        self.video_label = tk.Label(self.video_frame, text="Webcam Feed")
        self.video_label.grid(row=0, column=0, pady=5)
        
        # Create a button to capture an image
        self.capture_button = tk.Button(self.video_frame, text="Capture Image", command=self.capture_image)
        self.capture_button.grid(row=1, column=0, pady=5)
        
        # Create a label to display person's information
        self.info_label = tk.Label(self.info_frame, text="No person recognized", wraplength=300, justify="left", font=("Helvetica", 12))
        self.info_label.grid(row=0, column=0, pady=5)
        
        # Create a label and Listbox to display parties
        tk.Label(self.voting_frame, text="Choose a party to vote for:", font=("Helvetica", 12)).pack(pady=5)
        self.party_var = tk.StringVar(value=parties)
        self.party_listbox = tk.Listbox(self.voting_frame, listvariable=self.party_var, selectmode=tk.SINGLE, height=len(parties))
        self.party_listbox.pack(pady=5)
        
        # Create a button to submit the vote
        self.submit_button = tk.Button(self.voting_frame, text="Submit Vote", command=self.submit_vote, state=tk.DISABLED)
        self.submit_button.pack(pady=5)
        
        # Create a variable to store the captured image
        self.captured_image = None
        
        # Start displaying the video feed
        self.update_video_feed()
    
    def update_video_feed(self):
        # Capture a frame from the webcam
        ret, frame = self.cap.read()
        
        if ret:
            # Convert the frame from BGR to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert the frame to a PIL Image
            image = Image.fromarray(frame_rgb)
            
            # Resize the image (reduce to half size)
            resized_image = image.resize((image.width // 2, image.height // 2))
            
            # Convert the resized image to a PhotoImage and display it in the label
            image_tk = ImageTk.PhotoImage(resized_image)
            self.video_label.config(image=image_tk)
            self.video_label.image = image_tk
        
        # Schedule the next update of the video feed
        self.after(10, self.update_video_feed)
    
    def capture_image(self):
        # Capture a frame from the webcam
        ret, frame = self.cap.read()
        
        if ret:
            # Convert the frame from BGR to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Store the captured image
            self.captured_image = frame_rgb
            
            # Perform face recognition on the captured image
            self.recognize_face()
    
    def recognize_face(self):
        try:
            # Perform face recognition using DeepFace
            result = DeepFace.find(
                img_path=self.captured_image,
                db_path=db_path,
                distance_metric='euclidean_l2',
                model_name='Facenet512',
                detector_backend='yolov8'
            )
        except ValueError as e:
            messagebox.showerror("Error", f"Error during face detection: {e}")
            return

        # Connect to the SQLite database
        conn = sqlite3.connect(db_file_path)
        cur = conn.cursor()

        # Initialize found_match to False
        found_match = False
        
        # Check each DataFrame in the results
        if result:
            # Take the first recognized face only and stop further processing
            for df in result:
                if found_match:
                    break
                for _, row in df.iterrows():
                    # Get the file path of the recognized face
                    file_path = row['identity']
                    # Extract the name of the face from the file path
                    name = os.path.splitext(os.path.basename(file_path))[0]

                    # Query the person data from the SQLite database
                    cur.execute("SELECT * FROM person_data WHERE name=?", (name,))
                    person_data = cur.fetchone()
                    
                    if person_data:
                        # Display the person's information in the info_label
                        info_text = (f"Name: {person_data[0]}\n"
                                    f"ID: {person_data[1]}\n"
                                    f"Age: {person_data[2]}\n"
                                    f"Gender: {person_data[3]}")
                        self.info_label.config(text=info_text)
                        
                        # Check if the person has already voted
                        cur.execute("SELECT has_voted FROM voted_faces WHERE name=?", (name,))
                        voted_face = cur.fetchone()
                        
                        if voted_face:
                            # If the person has already voted, inform them and do not allow them to vote again
                            if voted_face[0] == 1:
                                messagebox.showinfo("Vote Status", f"{name} has already voted.")
                                self.info_label.config(text=f"{name} has already voted.")
                                self.submit_button.config(state=tk.DISABLED)
                            else:
                                # Allow the person to vote
                                messagebox.showinfo("Vote Status", f"{name}, you may vote.")
                                self.name = name
                                self.party_listbox.delete(0, tk.END)
                                self.party_listbox.insert(0, *parties)
                                self.submit_button.config(state=tk.NORMAL)
                            found_match = True
                        else:
                            # Insert the recognized face into the database and allow them to vote
                            image_bytes = open(file_path, 'rb').read()
                            cur.execute("INSERT INTO voted_faces (name, image, has_voted) VALUES (?, ?, 0)", 
                                        (name, image_bytes))
                            conn.commit()
                            messagebox.showinfo("Vote Status", f"{name}, you may vote.")
                            self.name = name
                            self.party_listbox.delete(0, tk.END)
                            self.party_listbox.insert(0, *parties)
                            self.submit_button.config(state=tk.NORMAL)
                            found_match = True
                        # If a match is found, stop processing
                        break
                    else:
                        # Person not found in the database
                        self.info_label.config(text="No additional information available.")
        
        # If no match is found, display a message
        if not found_match:
            messagebox.showinfo("Vote Status", "No vote allowed.")
            self.info_label.config(text="No additional information available.")
        
        # Close the database connection
        conn.close()

    
    def submit_vote(self):
        # Get the selected party
        selected_index = self.party_listbox.curselection()
        if selected_index:
            selected_party = parties[selected_index[0]]
            
            # Update the voted_faces table with the selected party and set has_voted to 1
            conn = sqlite3.connect(db_file_path)
            cur = conn.cursor()
            cur.execute("UPDATE voted_faces SET has_voted=1, party=? WHERE name=?", (selected_party, self.name))
            conn.commit()
            conn.close()
            
            # Disable the submit button
            self.submit_button.config(state=tk.DISABLED)
            
            # Notify the user that the vote was submitted
            messagebox.showinfo("Vote Submitted", f"Your vote for {selected_party} has been submitted.")
    
    def on_closing(self):
        # Release the webcam and close the tkinter window
        self.cap.release()
        self.destroy()

# Run the tkinter application
if __name__ == "__main__":
    # Create the SQLite database and load data from CSV
    create_database()
    
    # Create and run the application
    app = FaceRecognitionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()