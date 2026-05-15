import os
import sys
import pandas as pd
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

def upload_csv_to_mongodb(csv_file_path: str):
    """
    Reads a local greenhouse CSV file and uploads its contents 
    directly to the specified MongoDB Atlas collection.
    """
    
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    DATABASE_NAME = os.getenv("DATABASE_NAME")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")

    try:
        # 2. Check if file exists
        if not os.path.exists(csv_file_path):
            print(f"Error: The file path '{csv_file_path}' does not exist.")
            return

        print(f"Reading data from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # Clean data: MongoDB cannot natively ingest DataFrames, convert to dictionary list
        # handling NaN values by converting them to None (represented as null in Mongo)
        df_clean = df.replace({float('nan'): None})
        data_dict = df_clean.to_dict(orient="records")

        if not data_dict:
            print("Warning: The CSV file is empty. Nothing to upload.")
            return

        print(f"Parsed {len(data_dict)} records. Initializing MongoDB connection...")

        # 3. Establish secure connection using certifi to avoid SSL handshake errors
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        # 4. Insert data into the collection
        print(f"Uploading records to database '{DATABASE_NAME}', collection '{COLLECTION_NAME}'...")
        result = collection.insert_many(data_dict)
        
        print("🎉 Upload successful!")
        print(f"Inserted Document IDs Count: {len(result.inserted_ids)}")

    except Exception as e:
        print(f"An error occurred during the upload process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Provide the path to your raw or split data file here
    CSV_PATH = "artifacts/greenhouse_crop_yields.csv" 
    upload_csv_to_mongodb(CSV_PATH)
