import os
import shutil
import zipfile
import platform
import getpass

def get_kaggle_dir():
    system = platform.system()
    if system == 'Windows':
        return os.path.join('C:\\', 'Users', getpass.getuser(), '.kaggle')
    else:
        return os.path.join(os.path.expanduser("~"), ".kaggle")

def setup_kaggle_api(kaggle_json_path):
    kaggle_dir = get_kaggle_dir()
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(f"{kaggle_json_path} not found.")
    os.makedirs(kaggle_dir, exist_ok=True)
    shutil.copy(kaggle_json_path, kaggle_dir)
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
    print(f"Kaggle API setup complete. kaggle.json copied to {kaggle_dir}")

def download_and_extract_dataset(api, dataset, download_path, extract_path):
    api.dataset_download_files(dataset, path=download_path, unzip=False)
    print("Dataset downloaded successfully.")
    zip_file_path = os.path.join(download_path, f'{dataset.split("/")[-1]}.zip')
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Dataset unzipped successfully.")


current_dir = os.getcwd()
try:
    kaggle_json_path = os.path.join(current_dir,'util','kaggle.json')
    print(kaggle_json_path)
except:
    kaggle_json_path = os.path.join(current_dir,'util','kaggle.json')
    print(kaggle_json_path)


train_dir = os.path.join(current_dir, 'train')
dataset = 'clmentbisaillon/fake-and-real-news-dataset'

try:
    setup_kaggle_api(kaggle_json_path)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    download_and_extract_dataset(api, dataset, current_dir, train_dir)
except FileNotFoundError as e:
    print(f"File not found: {e}")
except zipfile.BadZipFile as e:
    print(f"Bad zip file: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

