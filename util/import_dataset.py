import os
import shutil
import zipfile
import platform
import getpass


def get_kaggle_dir():
    """Returns the path to the .kaggle directory based on the operating system."""
    system = platform.system()
    if system == 'Windows':
        return os.path.join('C:\\', 'Users', getpass.getuser(), '.kaggle')
    return os.path.join(os.path.expanduser("~"), ".kaggle")


def setup_kaggle_api(kaggle_json_path):
    """
    Sets up the Kaggle API by copying the kaggle.json file to the appropriate directory.

    Args:
        kaggle_json_path (str): The path to the kaggle.json file.

    Raises:
        FileNotFoundError: If the kaggle.json file is not found.
    """
    kaggle_dir = get_kaggle_dir()
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(f"{kaggle_json_path} not found.")

    os.makedirs(kaggle_dir, exist_ok=True)
    shutil.copy(kaggle_json_path, kaggle_dir)
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
    print(f"Kaggle API setup complete. kaggle.json copied to {kaggle_dir}")


def download_and_extract_dataset(api, dataset, download_path, extract_path):
    """
    Downloads and extracts a dataset from Kaggle.

    Args:
        api: An authenticated instance of KaggleApi.
        dataset (str): The name of the dataset to download.
        download_path (str): The path to download the dataset zip file.
        extract_path (str): The path to extract the dataset contents.
    """
    api.dataset_download_files(dataset, path=download_path, unzip=False)
    print("Dataset downloaded successfully.")

    zip_file_path = os.path.join(download_path, f'{dataset.split("/")[-1]}.zip')
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Dataset unzipped successfully.")


def import_dataset():
    current_dir = os.getcwd()
    kaggle_json_path = os.path.join(current_dir, 'kaggle_credential', 'kaggle.json')
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
