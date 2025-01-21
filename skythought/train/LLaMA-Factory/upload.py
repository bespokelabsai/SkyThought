from huggingface_hub import HfApi, Repository
import logging

# Enable logging to see what's happening
logging.basicConfig(level=logging.DEBUG)


# First, let's list what we're about to upload
import os
folder_path = "saves/Sky-T1-32B-Preview/full/test"
files = os.listdir(folder_path)
print(f"Found {len(files)} files to upload:")
for f in files:
        print(f"- {f}")

# Initialize the API object
api = HfApi()

# Replace with your Hub username and desired repository name
repo_id = "ryanmarten/Sky-T1-32B-Preview-5k-1-epoch"

# Push the model to the hub
api.upload_folder(
    folder_path="saves/Sky-T1-32B-Preview/full/test",
    repo_id=repo_id,
    repo_type="model",
    )