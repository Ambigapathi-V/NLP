import os
from dagshub import get_repo_bucket_client

# Initialize DagsHub bucket client
repo_name = "Ambigapathi-V/NLP"
bucket_name = "NLP"
local_directory = "data/"  # Local directory for uploading

# Get the DagsHub S3 client
s3 = get_repo_bucket_client(repo_name)

# Function to upload all files in a local directory
def upload_files(local_directory='data/', bucket_name='NLP'):
    # Iterate over all files in the local directory
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)  # Full path of the file
            remote_path = os.path.join('data/', file)  # Remote path in the bucket
            
            # Upload the file to DagsHub
            s3.upload_file(
                Bucket=bucket_name,  # Repository name
                Filename=local_path,  # Local file path
                Key=remote_path  # Remote path in the bucket
            )
            print(f"Uploaded {local_path} to {remote_path}")


# Function to download multiple files from the DagsHub bucket
def download_file(remote_file_path='data/dataset.zip', download_directory='downloads'):
    # Create the download directory if it doesn't exist
    os.makedirs(download_directory, exist_ok=True)
    
    # Local file path where the downloaded file will be saved
    local_file_path = os.path.join(download_directory, os.path.basename(remote_file_path))

    try:
        # Download the file from DagsHub
        s3.download_file(
            Bucket=bucket_name,  # Repository name (bucket name)
            Key=remote_file_path,  # Remote file path in the bucket
            Filename=local_file_path  # Local path where the file will be saved
        )
        print(f"Downloaded {remote_file_path} to {local_file_path}")
    
    except Exception as e:
        print(f"Failed to download {remote_file_path}. Error: {e}")
# Example usage
if __name__ == "__main__":
    # Upload files from the local directory to the DagsHub bucket
    upload_files()
    download_file()