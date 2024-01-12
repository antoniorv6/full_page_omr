import json
import requests
import cv2
from PIL import Image
import numpy as np
import torch
import rich.progress as progress
from io import BytesIO
from torch.utils.data import Dataset
from data_augmentation import augment, convert_to_tensor_format
from utils import check_and_retrieveVocabulary
import random

def download_image(image_url: str):
    response = requests.get(image_url)

    if response.status_code == 200:
        image_bytes = BytesIO(response.content)
        img = Image.open(image_bytes)
        img = img.convert("RGB")
        return np.array(img)

    return None

def draw_bounding_box(image, bounding_box, color=(0, 0, 255), thickness=2):
    # Draw a blue rectangle on the image
    cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), color, thickness)

def get_data_split_by_staves(path, base_folder, reduce_ratio=0.8):
    Y = []
    X = []
    BboxesY = []
    all_seqs = []
    with open(path) as txtpath:
        samples = txtpath.readlines()
        for sample in progress.track(samples):
            sample = sample.replace('\n', '')
            with open(f"{base_folder}/gt/{sample}", "r") as jsoncontent:
                data = json.load(jsoncontent)
                image = data["filename"]
                erase = False
                page_sequences = []
                bboxes = []
                for page in data["pages"]:
                    for region in page["regions"]:
                        gt_sequence = []
                        if region["type"] == "staff":
                            x0 = region['bounding_box']["fromX"] - page["bounding_box"]["fromX"]
                            y0 = region['bounding_box']["fromY"] - page["bounding_box"]["fromY"]
                            w = region['bounding_box']["toX"] - region['bounding_box']["fromY"]
                            h = region['bounding_box']["toY"] - region['bounding_box']["fromY"]

                            bboxes.append([x0, y0, w, h])
                            if "symbols" not in region:
                                erase = True
                            else:
                                for symbol in region["symbols"]:
                                    gt_sequence.append(f"{symbol['agnostic_symbol_type']}:{symbol['position_in_staff']}")
                                page_sequences.append(gt_sequence)
                                all_seqs.append(gt_sequence)

                    if erase == True:
                        gt_sequence = []
                        page_sequences = []
                        print(f"Skipping {image}")
                        break
                    
                    else:
                        Y.append(page_sequences)
                        img = cv2.imread(f"{base_folder}/img/{data['filename']}")
                        bb = page['bounding_box']
                        img = img[bb["fromY"]:bb["toY"], bb["fromX"]:bb["toX"]]
                        width = bb["toX"] - bb["fromX"]
                        if width > 600:
                            img = img[30:img.shape[0]-60, (img.shape[1]//2)+25:img.shape[1]-60]
                            
                        width = int(np.ceil(img.shape[1] * reduce_ratio))
                        height = int(np.ceil(img.shape[0] * reduce_ratio))
                        img = cv2.resize(img, (width, height))
                        X.append(img)
                        BboxesY.append(bboxes)
    return X, Y, all_seqs, BboxesY


class FPOMRDataset_SPAN(Dataset):
    def __init__(self, base_folder, partition_folder, ratio, augment=False):
        self.x, y, _, _ = get_data_split_by_staves(path=partition_folder, base_folder=base_folder, reduce_ratio=ratio)
        self.augment = augment
        self.y = []
        for sample in y:
            complete = []
            for line in sample:
                complete.extend(line)
            self.y.append(complete)

    def __len__(self):
        return len(self.x)
    
    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        max_image_width = max([img.shape[1] for img in x])
        max_image_height = max([img.shape[0] for img in x])

        X_train = np.zeros(shape=[len(x), 3, max_image_height, max_image_width], dtype=np.float32)
        L_train = np.zeros(shape=[len(x)])

        if self.augment:
            X_train = augment(x)
        else:
            X_train = convert_to_tensor_format(x)
        
        L_train = (X_train.shape[2] // 8) * (X_train.shape[1] // 32)
               
        Y_train = []
        T_train = []

        Y_train = [self.w2i[element] for element in y]

        T_train.append(len(y))

        return torch.tensor(X_train), torch.tensor(np.array(Y_train), dtype=torch.long), torch.tensor(L_train, dtype=torch.long), torch.tensor(np.array(T_train), dtype=torch.long)


class FPOMRDataset_VAN(Dataset):
    def __init__(self, base_folder, partition_folder, ratio, augment=False):
        self.x, self.y, self.seqs, self.bboxes = get_data_split_by_staves(path=partition_folder, base_folder=base_folder, reduce_ratio=ratio)
        self.augment = augment
        
    
    def __len__(self):
        return len(self.x)
    
    def get_length(self):
        return [len(sample) for sample in self.y]
    
    def get_bboxes(self, idx):
        return self.bboxes[idx]
    
    def get_sentences(self, paragraph, iterations, blank_token):    
        sequenceList = []
        for line in paragraph:
            sequenceList.append(line) 
        if len(sequenceList) < iterations:
            for _ in range(len(sequenceList), iterations):
                sequenceList.append([blank_token])

        return sequenceList
    
    def set_max_iterations(self, n_iterations):
        self.req_iterations = n_iterations
    
    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.augment:
            augmented = augment(x)
            X_train = augmented
            #cv2.imwrite("test.jpg", 255. * augmented.transpose(1,2,0))
        else:
           x = convert_to_tensor_format(x)
           X_train = x
        
        L_train = (X_train.shape[2] // 8)
               
        Y_train = []
        T_train = []
        divided_y_batch = []

        divided_y_batch = self.get_sentences(y, self.req_iterations, len(self.w2i))
        max_length = max([len(seq) for seq in divided_y_batch])
        Y_train = np.zeros((self.req_iterations, max_length))

        for j, _ in enumerate(Y_train):
            for plc, value in enumerate(divided_y_batch[j]):
                if value == len(self.w2i):
                    Y_train[j, plc] = value
                else:
                    Y_train[j, plc] = self.w2i[value]
           
            T_train.append(len(divided_y_batch[j]))

        return torch.tensor(X_train), torch.tensor(np.array(Y_train), dtype=torch.long), torch.tensor(np.array(L_train), dtype=torch.long), torch.tensor(np.array(T_train), dtype=torch.long)

def load_dataset(base_folder, fold, ratio):
    train_dataset = FPOMRDataset_VAN(base_folder=base_folder, partition_folder=f"{base_folder}/partitions/fold{fold}/train.txt", ratio=ratio, augment=True)
    val_dataset = FPOMRDataset_VAN(base_folder=base_folder, partition_folder=f"{base_folder}/partitions/fold{fold}/val.txt", ratio=ratio, augment=False)
    test_dataset = FPOMRDataset_VAN(base_folder=base_folder, partition_folder=f"{base_folder}/partitions/fold{fold}/test.txt", ratio=ratio)

    w2i, i2w = check_and_retrieveVocabulary([train_dataset.seqs, val_dataset.seqs, test_dataset.seqs], pathOfSequences='vocab/', nameOfVoc='SEILS')

    train_dataset.set_dictionaries(w2i, i2w)
    val_dataset.set_dictionaries(w2i, i2w)
    test_dataset.set_dictionaries(w2i, i2w)

    return train_dataset, val_dataset, test_dataset


def load_dataset_unfolding(base_folder, fold, ratio):
    train_dataset = FPOMRDataset_SPAN(base_folder=base_folder, partition_folder=f"{base_folder}/partitions/fold{fold}/train.txt", ratio=ratio, augment=True)
    val_dataset = FPOMRDataset_SPAN(base_folder=base_folder, partition_folder=f"{base_folder}/partitions/fold{fold}/val.txt", ratio=ratio, augment=False)
    test_dataset = FPOMRDataset_SPAN(base_folder=base_folder, partition_folder=f"{base_folder}/partitions/fold{fold}/test.txt", ratio=ratio)

    w2i, i2w = check_and_retrieveVocabulary([train_dataset.y, val_dataset.y, test_dataset.y], pathOfSequences='vocab/', nameOfVoc='SEILS')

    train_dataset.set_dictionaries(w2i, i2w)
    val_dataset.set_dictionaries(w2i, i2w)
    test_dataset.set_dictionaries(w2i, i2w)

    return train_dataset, val_dataset, test_dataset




