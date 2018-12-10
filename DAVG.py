import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from dataset import PostData, ExtractEmbeddings, ToTensor


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Model architecture
        self.fc1 = nn.Linear(300, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1003)

    def forward(self, input):
        out1 = self.fc1(input)
        out1 = F.relu(out1)
        out2 = self.fc2(out1)
        out2 = F.relu(out2)
        out3 = self.fc3(out2)
        probs = F.sigmoid(out3)
        return probs


class DAVG:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = Model().to(self.device)

    def train_model(self, data_file, embeddings_file, validation_split=0.2, batch_size=5, epochs=1,
                    learning_rate=1e-3):
        transformed_dataset = PostData(csv_file=data_file,
                                       transform=transforms.Compose([ExtractEmbeddings(embeddings_file), ToTensor()]))

        dataset_size = len(transformed_dataset)
        idxs = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        np.random.shuffle(idxs)

        train_idxs, val_idxs = idxs[split:], idxs[:split]
        train_sampler = SubsetRandomSampler(train_idxs)
        val_sampler = SubsetRandomSampler(val_idxs)

        train_dataloader = DataLoader(transformed_dataset, batch_size=batch_size, num_workers=2,
                                      sampler=train_sampler)
        val_dataloader = DataLoader(transformed_dataset, batch_size=batch_size, num_workers=2,
                                    sampler=val_sampler)

        mse_loss = torch.nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_train_loss = 0.0
            for i, batch_data in enumerate(train_dataloader):
                batch_X, batch_y = batch_data['body'].to(self.device), batch_data['labels'].to(self.device)

                optimizer.zero_grad()
                probs = self.model(batch_X)

                loss = mse_loss(probs, batch_y)
                epoch_train_loss += loss / batch_size

                loss.backward()
                optimizer.step()

                print('Iteration [%d] loss: %f' % (i, loss))

            print('[Training] Epoch loss %f' % epoch_train_loss)
            epoch_val_loss = 0.0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    batch_X, batch_y = batch_data['body'].to(self.device), batch_data['labels'].to(self.device)
                    probs = self.model(batch_X)
                    val_loss = mse_loss(probs, batch_y)

                    epoch_val_loss += val_loss / batch_size
                print('[Validation] Epoch loss %f' % epoch_val_loss)

            if epoch % 10 == 0 and epoch != 0:
                self.save_model()

    def test_model(self, test_data_file, embeddings_file, batch_size=5, threshold=0.5):
        transformed_dataset = PostData(csv_file=test_data_file,
                                       transform=transforms.Compose([ExtractEmbeddings(embeddings_file), ToTensor()]))
        dataset_size = len(transformed_dataset)
        dataloader = DataLoader(transformed_dataset, batch_size=batch_size, num_workers=2)
        with torch.no_grad():
            precision = 0.
            accuracy = 0.
            for i, batch_data in enumerate(dataloader):
                batch_X, batch_y = batch_data['body'].to(self.device), batch_data['labels']

                probs = self.model(batch_X)
                probs = probs.data.cpu().numpy()

                y_preds = np.where(probs < threshold, 0, 1)
                y_gt = batch_y.numpy()

                idxs = (y_gt == 1)

                precision += np.mean((y_preds[idxs] == y_gt[idxs]).astype(int))
                accuracy += np.mean((y_preds == y_gt).astype(int))

        print('Test Precision: ', precision / dataset_size)
        print('Test Accuracy: ', accuracy / dataset_size)

    def save_model(self):
        torch.save(self.model.state_dict(), './checkpoints/model_params.pt')

    def load_model(self):
        self.model = Model().to(self.device)
        self.model.load_state_dict(torch.load('./checkpoints/model_params.pt'))
        self.model.eval()


if __name__ == '__main__':
    model = DAVG()
    model.load_model()
    model.train_model(data_file='./data/baseline_train_data.csv', embeddings_file='./gensim_wordmodel')
    # model.test_model(test_data_file='./data/baseline_train_data.csv', embeddings_file='./gensim_wordmodel')
    # model.save_model()
