import os
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import clip
from PIL import Image, ImageFile
import torch.nn.functional as F

# if you changed the MLP architecture during training, change it also here:


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class AestheticPredictor:

    def __init__(self, device="cuda"):
        model = MLP(768)

        s = torch.load(
            "/fsx/users/haboutal/home/github/MultimodalDataset/src/model/vision_qa/sac+logos+ava1-l14-linearMSE.pth", map_location=device)

        model.load_state_dict(s)
        model.to(device)
        model.eval()

        model2, preprocess = clip.load("ViT-L/14", device=device)

        model_dict = {}
        model_dict['classifier'] = model
        model_dict['clip_model'] = model2
        model_dict['clip_preprocess'] = preprocess
        model_dict['device'] = device

        self.model_dict = model_dict

    def normalized(self, a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def predict(self, image):
        image_input = self.model_dict['clip_preprocess'](
            image).unsqueeze(0).to(self.model_dict['device'])
        with torch.no_grad():
            image_features = self.model_dict['clip_model'].encode_image(
                image_input)
            if self.model_dict['device'] == 'cuda':
                im_emb_arr = self.normalized(
                    image_features.detach().cpu().numpy())
                im_emb = torch.from_numpy(im_emb_arr).to(
                    self.model_dict['device']).type(torch.cuda.FloatTensor)
            else:
                im_emb_arr = self.normalized(image_features.detach().numpy())
                im_emb = torch.from_numpy(im_emb_arr).to(
                    self.model_dict['device']).type(torch.FloatTensor)

            prediction = self.model_dict['classifier'](im_emb)
        score = prediction.item()

        return score
