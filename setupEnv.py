import os.path
import requests

links = [
    ('RDD-v2.pth', 'https://univr-my.sharepoint.com/:u:/g/personal/enrico_bonoldi_studenti_univr_it/EWEQXe7du41KpG9L2sZGM8cByUAmy6TgICSa8p0aIbIRGw?e=LnWp5y&download=1'),
]

download_dir = 'download'

def download_file_if_not_exists(url, filename=None, download_folder='downloads'):
    # Create download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
        print(f"Created folder: {download_folder}")

    # Get the filename from the URL
    filename = filename if filename else os.path.basename(url)
    file_path = os.path.join(download_folder, filename)

    # Download file if it doesn't exist
    if not os.path.isfile(file_path):
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded and saved to: {file_path}")
        except requests.RequestException as e:
            print(f"Download failed: {e}")
    else:
        print(f"File already exists: {file_path}")

    return file_path


if __name__ ==  '__main__':
    print('Syncing...')
    for (filename, link) in links:
        download_file_if_not_exists(link,filename)
    
    