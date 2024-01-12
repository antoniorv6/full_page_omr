from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import lightning.pytorch as L
import torch.optim as optim
import random
from torchinfo import summary
from models.VAN import get_VAN_model
from models.SPAN import get_span_recurrent_model
from itertools import groupby
from utils import levenshtein

class VAN_Lightning(L.LightningModule):
    def __init__(self, i2w, max_iterations):
        super().__init__()
        self.model, _ = get_VAN_model(100, 256, 256, 256, 256, len(i2w)+1)
        self.i2w = i2w
        self.loss = nn.CTCLoss(blank=len(i2w), reduction='sum')
        self.max_iterations = max_iterations
        self.acc_ed = 0
        self.acc_len = 0
        self.random_sample = random.randint(0,10)
        self.save_hyperparameters()
        pass

    def forward(self, x):
        features = self.model.forward_encoder(x)
        prediction = self.model.forward_decoder_pass(features)
        return prediction

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.model.parameters(), lr=0.0001, amsgrad=False)
    
    def training_step(self, train_batch):
        x, y, l, t = train_batch
        self.model.reset_status_forward()
        global_loss = 0
        features = self.model.forward_encoder(x)
        for iteration in range(self.max_iterations):
            prediction = self.model.forward_decoder_pass(features)
            loss = self.loss(prediction.permute(2,0,1), y.squeeze(0)[iteration].unsqueeze(0), l, t.squeeze(0)[iteration].unsqueeze(0))
            global_loss += loss
                
        self.log("loss", global_loss, on_epoch=True, batch_size=1, prog_bar=True)
        
        return global_loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y, _, _ = val_batch
        self.model.reset_status_forward()
        features = self.model.forward_encoder(x)
        decoded = []
        for iteration in range(self.max_iterations):
            prediction = self.model.forward_decoder_pass(features)
            prediction = prediction.permute(0,2,1)
            prediction = prediction[0]
            out_best = torch.argmax(prediction, dim=1)
            out_best = [k for k, g in groupby(list(out_best))]
            line = []
            for c in out_best:
                if c < len(self.i2w):  # CTC Blank must be ignored
                    line.append(self.i2w[c.item()])
            decoded += line
        
        gt = []
        for line in y[0]:
            gt += [self.i2w[label.item()] for label in line if label.item() < len(self.i2w) and label.item() > 0]
        
        if batch_idx == self.random_sample:
            print(f"[PR]: {decoded}")
            print(f"[GT]: {gt}")

        self.acc_ed += levenshtein(decoded, gt)
        self.acc_len += len(gt)
    
    def test_step(self, test_batch, batch_idx) -> STEP_OUTPUT | None:
        self.validation_step(test_batch, batch_idx)
    
    def on_validation_epoch_end(self, name="val") -> None:
        SER = 100. * self.acc_ed / self.acc_len
        self.log(f"{name}_SER", SER, prog_bar=True)
        self.random_sample = random.randint(0,10)
        self.acc_ed = 0
        self.acc_len = 0

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end(name="test")
        self.random_sample = random.randint(0,10)

class SPAN_Lightning(L.LightningModule):
    def __init__(self, i2w):
        super().__init__()
        self.model, _ = get_span_recurrent_model(1418,922,3, out_size=len(i2w)+1)
        self.i2w = i2w
        self.loss = nn.CTCLoss(blank=len(i2w))
        self.acc_ed = 0
        self.acc_len = 0
        self.random_sample = random.randint(0,10)
        self.save_hyperparameters()
        pass

    def forward(self, x):
        prediction = self.model.forward(x)
        return prediction

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.model.parameters(), lr=0.0001, amsgrad=False)
    
    def training_step(self, train_batch):
        x, y, l, t = train_batch
        prediction = self.model.forward(x)
        loss = self.loss(prediction.permute(1,0,2), y, l, t)
                
        self.log("loss", loss, on_epoch=True, batch_size=1, prog_bar=True)
        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y, _, _ = val_batch

        prediction = self.model.forward(x)
        #prediction = prediction.permute(0,2,1)
        prediction = prediction[0]
        out_best = torch.argmax(prediction, dim=1)
        out_best = [k for k, g in groupby(list(out_best))]
        decoded = []
        for c in out_best:
            if c < len(self.i2w):  # CTC Blank must be ignored
                decoded.append(self.i2w[c.item()])
        
        gt = [self.i2w[label.item()] for label in y[0] if label.item() < len(self.i2w) and label.item() > 0]
        
        if batch_idx == self.random_sample:
            print(f"[PR]: {decoded}")
            print(f"[GT]: {gt}")

        self.acc_ed += levenshtein(decoded, gt)
        self.acc_len += len(gt)
    
    def test_step(self, test_batch, batch_idx) -> STEP_OUTPUT | None:
        self.validation_step(test_batch, batch_idx)
    
    def on_validation_epoch_end(self, name="val") -> None:
        SER = 100. * self.acc_ed / self.acc_len
        self.log(f"{name}_SER", SER, prog_bar=True)
        self.random_sample = random.randint(0,10)
        self.acc_ed = 0
        self.acc_len = 0

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end(name="test")
        self.random_sample = random.randint(0,10)

def get_SPAN(i2w):
    model = SPAN_Lightning(i2w=i2w)
    return model

def get_VAN(i2w, max_iterations):
    model = VAN_Lightning(i2w=i2w, max_iterations=max_iterations)
    return model

        



