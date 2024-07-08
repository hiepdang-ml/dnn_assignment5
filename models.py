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


class FeatureExtractor:

    def __init__(self) -> None:
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # get the feature extractor blocks, drop last FC layer
        self.__extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.__extractor.eval()

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4
        assert input.shape[1:] == (3, 224, 224)
        return self.__extractor(input)


class Predictor:

    def __init__(
        self, 
        feature_extractor: FeatureExtractor,
        classifier: BaseEstimator,
    ):
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def fit(self, train_dataset: Dataset) -> None:
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=len(train_dataset), 
            shuffle=False
        )
        images: torch.Tensor; labels: torch.Tensor; filenames: List[str]
        images, labels, filenames = next(iter(train_dataloader))
        
        train_features: np.ndarray = self.feature_extractor(input=images)
        self.classifier.fit(
            X=train_features.detach().cpu().numpy(), 
            y=labels.cpu().numpy()
        )

    def predict(self, test_dataset: Dataset) -> pd.DataFrame:
        test_dataloader = DataLoader(
            dataset=test_dataset, 
            batch_size=len(test_dataset), 
            shuffle=False
        )
        images: torch.Tensor; filenames: List[str]
        images, filenames = next(iter(test_dataloader))

        test_features: np.ndarray = self.feature_extractor(input=images)
        predicted_labels: np.ndarray = self.classifier.predict(
            X=test_features.detach().cpu().numpy()
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

    train_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Train')
    test_dataset = DogHearUnlabeledDataset(data_root='Dog_heart/Test')

    # SVM
    resnet_svm = Predictor(feature_extractor=FeatureExtractor(), classifier=SVC())
    resnet_svm.fit(train_dataset)
    resnet_svm.predict(test_dataset)

    # kNN with k=5
    resnet_5nn = Predictor(feature_extractor=FeatureExtractor(), classifier=KNeighborsClassifier(n_neighbors=5))
    resnet_5nn.fit(train_dataset)
    resnet_5nn.predict(test_dataset)

