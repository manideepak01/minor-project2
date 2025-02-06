import sqlite3
import pandas as pd

# Connect to the SQLite database (or create one if it doesn't exist)
conn = sqlite3.connect('faces/voter_database.db')

# Create a cursor object
cur = conn.cursor()

# Create tables for storing person data and images
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
        party TEXT
    )
''')

# Load data from CSV into person_data table
data = pd.read_csv('faces/data.csv')
data.to_sql('person_data', conn, if_exists='replace', index=False)

# Commit changes and close the connection
conn.commit()
conn.close()
