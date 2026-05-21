from huggingface_hub import HfApi, login

# Set your token as environment variable: $env:HF_TOKEN="your_token"
import os
token = os.environ.get("HF_TOKEN", "")
login(token=token)

api = HfApi()

repo_id = "Sharvari22/layoutlmv3-funsd"

api.upload_folder(
    folder_path="models/layoutlmv3-finetuned",
    repo_id=repo_id,
    repo_type="model"
)

print(f"Model uploaded to: https://huggingface.co/{repo_id}")