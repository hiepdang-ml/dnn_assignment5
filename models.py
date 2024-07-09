from typing import List

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from datasets import DogHeartLabeledDataset, DogHearUnlabeledDataset


class FeatureExtractor(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # get the feature extractor blocks, drop last FC layer
        self.__extractor = nn.Sequential(
            *list(resnet.children())[:-1], 
            nn.Flatten(start_dim=1, end_dim=-1)
        ).half()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4
        assert input.shape[1:] == (3, 224, 224)
        return self.__extractor(input)

class Predictor:

    def __init__(
        self, 
        feature_extractor: nn.Module,
        classifier: BaseEstimator,
        device: torch.device = torch.device('cuda')
    ):
        self.feature_extractor = feature_extractor.to(device)
        self.classifier = classifier
        self.device = device

    def fit(self, train_dataset: Dataset) -> None:
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=32, 
            shuffle=True,
        )
        input_tensors: List[torch.Tensor] = []
        label_tensors: List[torch.Tensor] = []
        for input_tensor, label_tensor, _ in train_dataloader:
            input_tensors.append(input_tensor.to(device=self.device, dtype=torch.half))
            label_tensors.append(label_tensor.to(device=self.device))

        input_tensor: torch.Tensor = torch.cat(input_tensors)
        label_tensor: torch.Tensor = torch.cat(label_tensors)

        with torch.no_grad():
            self.feature_extractor.eval()
            train_features: np.ndarray = self.feature_extractor(input=input_tensor)

        self.classifier.fit(
            X=train_features.detach().cpu().numpy(), 
            y=label_tensor.cpu().numpy()
        )

    def predict(self, test_dataset: Dataset) -> pd.DataFrame:
        test_dataloader = DataLoader(
            dataset=test_dataset, 
            batch_size=32, 
            shuffle=False
        )
        input_tensors: List[torch.Tensor] = []
        filenames: List[str] = []
        for input_tensor, fnames in test_dataloader:
            input_tensors.append(input_tensor.to(device=self.device, dtype=torch.half))
            filenames.extend(fnames)

        input_tensor: torch.Tensor = torch.cat(input_tensors)

        with torch.no_grad():
            self.feature_extractor.eval()
            test_features: np.ndarray = self.feature_extractor(input=input_tensor)

        predicted_labels: np.ndarray = self.classifier.predict(
            X=test_features.detach().cpu().numpy(),
        )
        prediction_table = pd.DataFrame(
            data={'image': filenames, 'label': predicted_labels}
        )
        prediction_table.to_csv(
            f'{self.classifier.__class__.__name__}_'
            f'{self.feature_extractor.__class__.__name__}.csv', 
            header=False, 
            index=False,
        )
        return prediction_table


if __name__ == '__main__':

    device = torch.device('cuda')

    train_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Train')
    test_dataset = DogHearUnlabeledDataset(data_root='Dog_heart/Test')

    # SVM
    resnet_svm = Predictor(
        feature_extractor=FeatureExtractor(), 
        classifier=SVC(), 
        device=device, 
    )
    resnet_svm.fit(train_dataset)
    resnet_svm.predict(test_dataset)

    # kNN with k=5
    resnet_5nn = Predictor(
        feature_extractor=FeatureExtractor(), 
        classifier=KNeighborsClassifier(n_neighbors=5),
        device=device, 
    )
    resnet_5nn.fit(train_dataset)
    resnet_5nn.predict(test_dataset)
