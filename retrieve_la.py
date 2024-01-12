from data import load_dataset
from torch.utils.data import DataLoader
from itertools import groupby
from ModelManager import get_VAN, VAN_Lightning
import torch
import cv2
import numpy as np
import os

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # compute the area of intersection rectangle
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    interArea = inter_width * inter_height

    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def find_bounding_box(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box for each contour
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    return bounding_boxes[0]

def draw_bounding_box(image, bounding_box, color=(0, 0, 255), thickness=2):
    # Draw a blue rectangle on the image
    cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), color, thickness)

def overlay_mask(image, mask):
    # Create a copy of the original image
    overlay = image.copy()

    # Set the red channel to 255 in the overlay where the mask is 1
    overlay[:, :, 2] = np.where(mask.squeeze() == 1, 255, overlay[:, :, 2])

    return overlay

def crop_expand_image(image, crop_top=30, expand_bottom=30):
    # Crop the top of the image
    #print(image.shape)
    cropped_image = image[crop_top:, :]

    # Create an expanded bottom by adding zeros
    expanded_bottom = np.zeros((expand_bottom, image.shape[1]), dtype=np.uint8)

    # Stack the cropped image and expanded bottom
    result = np.vstack([cropped_image, expanded_bottom])

    return result

def main():
    train_ds, val_ds, test_ds = load_dataset(base_folder="Data/SEILS", fold=0, ratio=1.0)

    req_iterations = max(train_ds.get_length() + val_ds.get_length() + test_ds.get_length())

    train_ds.set_max_iterations(req_iterations)
    val_ds.set_max_iterations(req_iterations)  
    test_ds.set_max_iterations(req_iterations)  

    train_dataloader = DataLoader(train_ds, batch_size=1, num_workers=20)
    val_dataloader = DataLoader(val_ds, batch_size=1, num_workers=20)
    test_dataloader = DataLoader(test_ds, batch_size=1, num_workers=20)

    model = VAN_Lightning.load_from_checkpoint("weights/SEILS/VAN_fold_0.ckpt").model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        idx = 0
        for batch in test_dataloader:
            os.makedirs(f"attention_vis/{idx}", exist_ok=True)
            x, y, _, _ = batch
            save_img = x.squeeze(0).permute(1,2,0).cpu().detach().numpy()
            bboxing = save_img.copy() * 255.
            cv2.imwrite("img.png", save_img * 255.)
            model.reset_status_forward()
            features = model.forward_encoder(x.to(device))
            decoded = []
            gt_bb = test_ds.get_bboxes(idx)
            for iteration in range(req_iterations):
                prediction = model.forward_decoder_pass(features)
                prediction = prediction.permute(0,2,1)
                prediction = prediction[0]
                out_best = torch.argmax(prediction, dim=1)
                out_best = [k for k, g in groupby(list(out_best))]
                line = []
                for c in out_best:
                    if c < len(test_ds.i2w):  # CTC Blank must be ignored
                        line.append(test_ds.i2w[c.item()])
                
                if len(line) > 0:
                    line.append('\n')
                    decoded += line

                    attention_weights = model.get_attention_weights()#.unsqueeze(2).expand(-1, -1, features.size(-1))
                    attention_weights = torch.where(attention_weights <= 0.25, torch.tensor(0.0), torch.tensor(1.0))
                    attention_weights = attention_weights.unsqueeze(2).expand(-1, -1, features.size(-1)).squeeze(0)
                    #min_vals, _ = torch.min(attention_weights, dim=1, keepdim=True)
                    #max_vals, _ = torch.max(attention_weights, dim=1, keepdim=True)
                    #attention_weights = (attention_weights - min_vals) / (max_vals - min_vals)
                    attention_img = attention_weights.cpu().detach().numpy()
                    attention_img = cv2.resize(attention_img, (x.shape[3], x.shape[2]), interpolation=cv2.INTER_NEAREST)
                    attention_img = crop_expand_image(attention_img)
                    result = overlay_mask(save_img*255., attention_img)

                    bbox = list(find_bounding_box(attention_img))
                    gt_bbox = gt_bb[iteration]
                    bbox[0] = 10
                    gt_bbox[0] = bbox[0]
                    gt_bbox[2] = bbox[2]
                    print(bb_intersection_over_union(gt_bbox, bbox))

                    draw_bounding_box(bboxing, bbox)
                    #print(attention_img.shape)
                    cv2.imwrite(f"attention_vis/{idx}/{iteration}.png", result)
                    #print(attention_weights.size())
                    #print(x.size())
                    #print(attention_weights)
            
            cv2.imwrite(f"attention_vis/{idx}/bboxes.png", bboxing)
            import sys
            sys.exit()

            idx += 1


if __name__ == "__main__":
    main()