import os
import sys
import requests
import zipfile
import shutil

DATA_DIR = os.path.join(os.getcwd(), 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VAL_DIR = os.path.join(DATA_DIR, 'val')

def download_url(url, path, extract_path, zip=False):
    """
    Download a file from a url and save it to a path
    :param url: url to download from
    :param path: path to save the file
    :param extract_path: path to extract the file to
    :param zip: whether the file is a zip file
    :return: None
    """
    r = requests.get(url)
    with open(path, 'wb') as f:
        f.write(r.content)
    if zip:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)


def move_files(source, destination):
    allfiles = os.listdir(source)
    
    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        print(dst_path)
        os.rename(src_path, dst_path)

    shutil.rmtree(source)

# create dirs
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)
if not os.path.exists(VAL_DIR):
    os.makedirs(VAL_DIR)


# download test
print('Downloading test data...')
test_url = 'https://zenodo.org/record/3993380/files/ribfrac-test-images.zip?download=1'
file_path = os.path.join(TEST_DIR, 'ribfrac-test-images.zip')
download_url(test_url, file_path, TEST_DIR, zip=True)
# delete zip
os.remove(file_path)
print('Test data done!')

# download validation
print('Downloading validation data...')
# download images
val_images_url = 'https://zenodo.org/record/3893496/files/ribfrac-val-images.zip?download=1'
file_path = os.path.join(VAL_DIR, 'ribfrac-val-images.zip')
download_url(val_images_url, file_path, VAL_DIR, zip=True)
# download labels
val_labels_url = 'https://zenodo.org/record/3893496/files/ribfrac-val-labels.zip?download=1'
file_path = os.path.join(VAL_DIR, 'ribfrac-val-labels.zip')
download_url(val_labels_url, file_path, VAL_DIR, zip=True)
# delete zip
os.remove(file_path)
# download info
val_info_url = 'https://zenodo.org/record/3893496/files/ribfrac-val-info.csv?download=1'
download_url(val_info_url, os.path.join(VAL_DIR, 'ribfrac-val-info.csv'), VAL_DIR, zip=False)
print('Validation data done!')

# download train
print('Downloading train data...')
# download train 1
# download images 1
train_images_url = 'https://zenodo.org/record/3893508/files/ribfrac-train-images-1.zip?download=1'
file_path = os.path.join(TRAIN_DIR, 'ribfrac-train-images-1.zip')
download_url(train_images_url, file_path, TRAIN_DIR, zip=True)
os.remove(file_path)


# download train 2
# download images 1
train_images_url = 'https://zenodo.org/record/3893498/files/ribfrac-train-images-2.zip?download=1'
file_path = os.path.join(TRAIN_DIR, 'ribfrac-train-images-2.zip')
download_url(train_images_url, file_path, TRAIN_DIR, zip=True)
os.remove(file_path)


# merge train 1 and train 2
# move images
train_images_1 = os.path.join(TRAIN_DIR, 'Part1')
train_images_2 = os.path.join(TRAIN_DIR, 'Part2')
train_images = os.path.join(TRAIN_DIR, 'ribfrac-train-images')
os.makedirs(train_images)
move_files(train_images_1, train_images)
move_files(train_images_2, train_images)

# download labels 1
train_labels_url = 'https://zenodo.org/record/3893508/files/ribfrac-train-labels-1.zip?download=1'
file_path = os.path.join(TRAIN_DIR, 'ribfrac-train-labels-1.zip')
download_url(train_labels_url, file_path, TRAIN_DIR, zip=True)
os.remove(file_path)

# download labels 2
train_labels_url = 'https://zenodo.org/record/3893498/files/ribfrac-train-labels-2.zip?download=1'
file_path = os.path.join(TRAIN_DIR, 'ribfrac-train-labels-2.zip')
download_url(train_labels_url, file_path, TRAIN_DIR, zip=True)
os.remove(file_path)


# move labels
train_labels_1 = os.path.join(TRAIN_DIR, 'Part1')
train_labels_2 = os.path.join(TRAIN_DIR, 'Part2')
train_labels = os.path.join(TRAIN_DIR, 'ribfrac-train-labels')
os.makedirs(train_labels)
move_files(train_labels_1, train_labels)
move_files(train_labels_2, train_labels)