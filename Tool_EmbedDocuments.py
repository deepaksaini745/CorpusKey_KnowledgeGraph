import os
import time
import pandas as pd
import numpy as np
import openai

# Read API key from file
api_key_file = r"input/APIkey.txt"
with open(api_key_file, "r") as f:
    api_key = f.read().strip()

openai.api_key = api_key #DS
# client = openai.OpenAI(api_key=api_key)

embeddingmodel = "text-embedding-3-small"

token_use_dict = {}
# Now you can use the 'client' object to interact with OpenAI API

# creates a dictionary of preferred models.  Note that the primary path entry with lower case is the text which is used in the OpenAi call
model_info = {
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "use case": "text",
        "input_cost": .5//1000000,
        "output_cost": 1.5//1000000,
        "markup": 10,
        "parameters": {
            "num_layers": 48,
            "num_heads": 12,
            "hidden_size": 4096,
            "context_window": 16385
        }
    },
    "gpt-4-turbo-preview": {
        "name": "GPT-4 Turbo Preview",
        "use case": "text",
        "input_cost": 10//1000000,
        "output_cost": 30//1000000,
        "markup": 3,
        "parameters": {
            "num_layers": 64,
            "num_heads": 16,
            "hidden_size": 8192,
            "context_window": 128000
        }
    },
    "gpt-4": {
        "name": "GPT-4 Turbo Preview",
        "use case": "text_complex",
        "input_cost": 30//1000000,
        "output_cost": 60//1000000,
        "markup": 3,
        "parameters": {
            "num_layers": 64,
            "num_heads": 16,
            "hidden_size": 8192,
            "context_window": 8192
        }},
    "gpt-3.5-turbo-0125": {
        "name": "GPT 3.5 Turbo 0125",
        "use case": "formatting",
        "input_cost": .50//1000000,
        "output_cost": 1.5//1000000,
        "markup": 10,
        "parameters": {
            "num_layers": None,
            "num_heads": None,
            "hidden_size": None,
            "context_window": 16385
        }
    },
    "gpt-3.5-turbo-1106": {
        "name": "GPT-3.5 Turbo 1106",
        "use case": "JSON",
        "input_cost": .5//1000000,
        "output_cost": 1.5//1000000,
        "markup": 10 ,
        "parameters": {
            "num_layers": None,
            "num_heads": None,
            "hidden_size": None,
            "context_window": 16385
        }
    },
    "text-embedding-3-small": {
        "name": "text-embedding-3-small",
        "use case": "embedding",
        "context_window": 16385,
        "input_cost": .02//1000000,
        "markup": 10,
        "parameters": {
            "num_layers": None,
            "num_heads": None,
            "hidden_size": None
        }
    }
    # Add more models as needed
}

def calculate_cost(token_use_dict, model_info):
    total_cost = 0
    
    for model, token_info in token_use_dict.items():
        if model in model_info:
            prompt_tokens = token_info.get('prompt_tokens', 0)
            completion_tokens = token_info.get('completion_tokens', 0)
            embed_tokens = token_info.get('embed_tokens', 0)
            prompt_cost = prompt_tokens * model_info[model]['input_cost'] * model_info[model]['markup']
            completion_cost = completion_tokens * model_info[model]['input_cost'] * model_info[model]['markup']
            embed_cost = embed_tokens * model_info[model]['input_cost'] * model_info[model]['markup']
            total_cost += prompt_cost + completion_cost + embed_cost
    
    return total_cost

def make_embeddings_folder(corpa_directory):
    output_folder = os.path.join(corpa_directory, "embeddings")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

# load user settings and api key
def read_settings(file_name):
    settings = {}
    with open(file_name, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            settings[key] = value
    return settings


    
# Define the function to send a batch of input text to the OpenAI API and return the embeddings
def embed_input_text(embed_string):
    global token_use_dict
    global embeddingmodel

    # response = client.embeddings.create( 
    response = openai.Embedding.create( #DS
        model=embeddingmodel,
        input=embed_string
    )
    print("working")
    embeddings = response.data
    embed_tokens = response.usage.total_tokens
    model = response.model

    for model in token_use_dict:
        if 'embed_tokens' not in token_use_dict[model]:
            token_use_dict[model]['embed_tokens'] = 0

        token_use_dict[model]['embed_tokens'] += embed_tokens
    else:
        if model not in token_use_dict:
                token_use_dict[model] ={}
        
        token_use_dict[model] = {'embed_tokens': embed_tokens}
    # Calculate total tokens

    return [response.embedding for response in embeddings]


def make_embeddings_array(textchunks_folder, embeddings_folder):
    for file in os.listdir(textchunks_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(textchunks_folder, file)
            df_chunks = pd.read_csv(file_path, encoding='utf-8', escapechar='\\')
            print(f"Loaded: {file_path}")
            # Embed the input text in batches of no more than MAX_TOKENS_PER_BATCH tokens each
            input_text_list = df_chunks.iloc[:, 1].tolist()
            embeddings = []
            for embed_string in input_text_list:
                embeddings.append(embed_input_text(embed_string))
            # Convert embeddings list to numpy array
            embeddings_array = np.array(embeddings)
            # Reshape embeddings array to remove the first dimension
            embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
            # Save the embeddings_array to the output_folder subfolder
            # Remove the file extension from the filename
            filename_without_extension = os.path.splitext(file)[0]
            npy_filename = f"{filename_without_extension}.npy"
            output_path = os.path.join(embeddings_folder, npy_filename)
            np.save(output_path, embeddings_array)




if __name__ == "__main__":
    print("This is module.py being run directly.")
    os.chdir(r"/Users/deepaksaini/Desktop/git/")
    current_dir = os.getcwd()

    corpa_directory = os.path.join(current_dir, "visualize_embeddings")

    # Define the maximum number of tokens per batch to send to OpenAI for embedding per minute
    MAX_TOKENS_PER_BATCH = 250000
    settings_file_path = os.path.join(corpa_directory, "input", "settings.txt")
    settings = read_settings(settings_file_path)
    
    # Load text data from Textchunks
    textchunks_folder = os.path.join(current_dir, "visualize_embeddings", "chunks")
    embeddings_folder = make_embeddings_folder(corpa_directory)
    make_embeddings_array(textchunks_folder, embeddings_folder)

    # make_embeddings_array("/Users/deepaksaini/Desktop/git/visualize_embeddings/output/test_chunks_csv/", 
    #                       "/Users/deepaksaini/Desktop/git/visualize_embeddings/output/test_chunks_npy/")