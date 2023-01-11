from torchvision import transforms

import gradio as gr
import numpy as np
import pytorch_lightning as pl
import torchmetrics as tm
import torch.nn as nn
import torch


class Lumiere(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = tm.Accuracy(task = 'multiclass', num_classes = 10)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, y)
        acc = self.accuracy(x_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    
    def predict(self, x):
        pred = self(x)
        return pred.argmax(1).item()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min = 1e-6)
        return [optimizer], [lr_scheduler]


model = Lumiere.load_from_checkpoint("/home/utilisateur/createch/mnistforweb/lightning_logs/version_6/checkpoints/epoch=9-step=8590.ckpt")
transform = transforms.Compose([transforms.ToTensor()])
def recognize_digit(image):
    img_3d=transform(image).unsqueeze(0)
    print(img_3d.shape)
    im_resize=img_3d/255.0
    prediction = model.predict(im_resize)
    return str(prediction)


gr.Interface(fn=recognize_digit, 
             inputs="sketchpad",
             outputs="label",
             live=True).launch() 