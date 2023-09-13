from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import numpy as np
import pdb
import torch

class FIDModel(torch.nn.Module):
    def __init__(self, dims=2048, to_device=True):
        super().__init__()
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx])
        self.pred_arr1 = []
        self.pred_arr2 = []
        self.model.eval()
        self.to_device = to_device
        self.device = next(self.model.parameters()).device

    def get_activations(self, batch, to_numpy=True):
        if self.to_device:
            batch = batch.to(self.device)
        with torch.no_grad():
            pred = self.model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        if to_numpy:
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        else:
            pred = pred.squeeze(3).squeeze(2)
        return pred

    def add_sample(self, sample1, sample2):
        pred1 = self.get_activations(sample1)
        pred2 = self.get_activations(sample2)
        self.pred_arr1.append(pred1)
        self.pred_arr2.append(pred2)

    def calculate_activation_statistics(self):
        pred_arr1 = np.concatenate(self.pred_arr1)
        mu1 = np.mean(pred_arr1, axis=0)
        sigma1 = np.cov(pred_arr1, rowvar=False)

        pred_arr2 = np.concatenate(self.pred_arr2)
        mu2 = np.mean(pred_arr2, axis=0)
        sigma2 = np.cov(pred_arr2, rowvar=False)
        fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        self.pred_arr1 = []
        self.pred_arr2 = []
        return fid_value

    def cals(self, pred_arr1, pred_arr2):
        pred_arr1 = np.concatenate(pred_arr1)
        mu1 = np.mean(pred_arr1, axis=0)
        sigma1 = np.cov(pred_arr1, rowvar=False)

        pred_arr2 = np.concatenate(pred_arr2)
        mu2 = np.mean(pred_arr2, axis=0)
        sigma2 = np.cov(pred_arr2, rowvar=False)
        fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value
