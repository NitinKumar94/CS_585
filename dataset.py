import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()


class PostData(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with post information.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file, header=0, index_col=False)
        self.data = self.data.head(100000)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx, :]
        sample = {
            'body': record['Body'],
            'labels': np.array(record[2:])  # Labels for presence or absence of tags
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ExtractEmbeddings:
    def __init__(self, model_file):
        self.model = Word2Vec.load(model_file)

    def __call__(self, sample):
        post = sample['body']
        post = post.lower()
        body = re.findall(r"[A-Za-z]+|\S", post)
        stop_free = " ".join([i for i in body if i not in stop])
        normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
        normalized = re.sub(r"\b\d+\b", "NUM", normalized)
        normalized = re.sub(r"\b[a-zA-Z]\b", "CHAR", normalized)
        tokens = normalized.split()

        embeddings = []
        for token in tokens:
            try:
                word_vec = self.model.wv[token]
                embeddings.append(word_vec)
            except KeyError:  # Ignoring tokens for which no embeddings were found
                pass

        embeddings = np.array(embeddings)

        sample['body'] = np.mean(embeddings, axis=0)

        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        body, labels = sample['body'], sample['labels'].astype('float32')
        return {'body': torch.from_numpy(body), 'labels': torch.from_numpy(labels)}


if __name__ == '__main__':
    transformed_dataset = PostData(csv_file='./data/baseline_train_data.csv',
                                   transform=transforms.Compose([ExtractEmbeddings('./gensim_wordmodel'), ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=100, shuffle=True, num_workers=2)

    for idx, batch_data in enumerate(dataloader):
        embeddings = batch_data['body']
        labels = batch_data['labels']
        print('X shape', embeddings.shape)
        print('labels shape', embeddings.shape)

