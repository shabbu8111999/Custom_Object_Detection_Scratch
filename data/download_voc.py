import os
import urllib.request
import tarfile


def download_and_extract_voc():
    try:
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
        save_path = "VOCtrainval_06-Nov-2007.tar"

        if not os.path.exists(save_path):
            print(f"Downloading PASCAL VOC 2007 dataset...")
            urllib.request.urlretrieve(url, save_path)

        print("Extracting dataset...")
        with tarfile.open(save_path) as tar:
            tar.extractall()

        print("Dataset downloaded and extracted successfully.")

    except Exception as e:
        print("Error While downlaoding the dataset:", e)

if __name__ == "__main__":
    download_and_extract_voc()