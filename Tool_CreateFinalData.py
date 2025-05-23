import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# project_folders now = ["input", "chunks", "embeddings", "bundle", "output"]

def create_final_data(directory, chunks_folder, embeddings_folder):
    # Get the sorted list of CSV and .npy files
    csv_files = sorted([f for f in os.listdir(chunks_folder) if f.endswith('.csv')])
    npy_files = sorted([f for f in os.listdir(embeddings_folder) if f.endswith('.npy')])

    # Initialize empty DataFrame and NumPy array for concatenation
    concatenated_csv = pd.DataFrame()
    concatenated_npy = None

    for csv_file, npy_file in zip(csv_files, npy_files):
        print(npy_file)
        # Read the CSV file and concatenate
        csv_path = os.path.join(chunks_folder, csv_file)
        csv_data = pd.read_csv(csv_path, encoding='utf-8', escapechar='\\')
        concatenated_csv = pd.concat([concatenated_csv, csv_data], ignore_index=True)

        npy_path = os.path.join(embeddings_folder, npy_file)
        npy_data = np.load(npy_path)
        if concatenated_npy is None:
            concatenated_npy = npy_data
        else:
            concatenated_npy = np.concatenate([concatenated_npy, npy_data], axis=0)


    # Save the concatenated data to the base folder
    output_csv = os.path.join(directory, "bundle", "chunks-originaltext.csv")
    concatenated_csv.to_csv(output_csv, encoding='utf-8', escapechar='\\', index=False)
    output_npy = os.path.join(directory, "bundle", "chunks.npy")
    np.save(output_npy, concatenated_npy)
    print("Files saved: chunks-originaltext.csv and chunks.npy")
    # Print the dimensions of the concatenated files
    print(f"chunks-originaltext.csv dimensions: {concatenated_csv.shape}")
    print(f"chunks.npy dimensions: {concatenated_npy.shape}")


def create_final_bundle(directory, chunks_folder):
    # Get the sorted list of CSV and .npy files
    csv_files = sorted([f for f in os.listdir(chunks_folder) if f.endswith('.csv')])

    # Initialize empty DataFrame and NumPy array for concatenation
    concatenated_csv = pd.DataFrame()

    for csv_file in csv_files:
        # Read the CSV file and concatenate
        csv_path = os.path.join(chunks_folder, csv_file)
        csv_data = pd.read_csv(csv_path, encoding='utf-8', escapechar='\\')
        concatenated_csv = pd.concat([concatenated_csv, csv_data], ignore_index=True)

    # Save the concatenated data to the base folder
    output_csv = os.path.join(directory, "bundle", "chunks-originaltext.csv")
    output_npy = os.path.join(directory, "bundle", "chunks.npy")

    concatenated_csv.to_csv(output_csv, encoding='utf-8', escapechar='\\', index=False)
    print("Files saved: chunks-originaltext.csv")
    # Print the dimensions of the concatenated csv file
    print(f"chunks-originaltext.csv dimensions: {concatenated_csv.shape}")

    # Load bundled csv data
    df = pd.read_csv(output_csv)

    # Extract text from chunks
    text_chunks = df['Text'].tolist()

    # SentenceTransformer for embeddings
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('all-mpnet-base-v2') # more descriptive but takes more time

    embeddings = model.encode(text_chunks, convert_to_tensor=True)
    embeddings_cpu = embeddings.cpu().numpy()

    np.save(output_npy, embeddings_cpu)
    print("Files saved: chunks.npy")
    # Print the dimensions of the concatenated csv file
    print(f"chunks.npy dimensions: {embeddings_cpu.shape}")



if __name__ == "__main__":
    print("This is module.py being run directly.")

    os.chdir(r"/Users/deepaksaini/Desktop/git/visualize_embeddings/")
    current_dir = os.getcwd()
    directory = os.path.join(current_dir)
    chunks_folder = os.path.join(directory, "chunks")
    embeddings_folder = os.path.join(directory, "embeddings")

    # create_final_data(directory, chunks_folder, embeddings_folder)
    create_final_bundle(directory, chunks_folder)
