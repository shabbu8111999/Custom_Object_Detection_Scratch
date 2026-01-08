import os
import tarfile
import urllib.request


def download_and_extract_voc():
    url = "https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    tar_path = "VOCtrainval_06-Nov-2007.tar"
    extract_path = "VOC2007"

    try:
        print("Downloading PASCAL VOC 2007 dataset...")

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        request = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(request) as response, open(tar_path, "wb") as out_file:
            out_file.write(response.read())

        size_mb = os.path.getsize(tar_path) / (1024 * 1024)
        print(f"Downloaded file size: {size_mb:.2f} MB")

        if size_mb < 400:
            raise ValueError("Download incomplete or blocked.")

        print("Extracting dataset...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_path)

        print("Dataset downloaded and extracted successfully.")

    except Exception as e:
        print("Error occurred:")
        print(e)


if __name__ == "__main__":
    download_and_extract_voc()
