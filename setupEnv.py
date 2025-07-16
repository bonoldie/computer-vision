import os
import requests
import shutil

from pathlib import Path

links = [
    ('RDD-v2.pth', 'https://univr-my.sharepoint.com/:u:/g/personal/enrico_bonoldi_studenti_univr_it/EWEQXe7du41KpG9L2sZGM8cByUAmy6TgICSa8p0aIbIRGw?e=LnWp5y&download=1'),
    ('MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', 'https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
    # ('dante_dataset.zip', 'https://univr-my.sharepoint.com/:u:/g/personal/enrico_bonoldi_studenti_univr_it/EaWhfmHAH9RKue-tR2JDvtEBIri9Q3Wn1hhFHJdSNUF42A?e=cH5ELe&download=1'),
]

download_dir = 'download'


def is_archive(extension):
    extension = extension.lower()
    for _, extensions, _ in shutil.get_unpack_formats():
        if extension in extensions:
            return True
    return False

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

            if(is_archive(os.path.splitext(filename)[1])):
                try:
                    archive_outdir = os.path.join(download_folder, os.path.splitext(filename)[0])
                    Path(archive_outdir).mkdir(parents=True, exist_ok=True)
                    print(f"Unpacking archive in: {archive_outdir}")

                    shutil.unpack_archive(file_path, archive_outdir)
                    print(f"Unpacking complete")
                except e:
                    print(f"Error while upacking downloaded archive: {e}")
                    pass
                       
        except requests.RequestException as e:
            print(f"Download failed: {e}")
    else:
        print(f"File already exists: {file_path}")

    return file_path


if __name__ ==  '__main__':
    print('Synching...')
    for (filename, link) in links:
        download_file_if_not_exists(link,filename)


    
    