import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from dataset import PostData, ExtractEmbeddings, ToTensor
import pickle
import time


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
        probs = torch.sigmoid(out3)
        return probs


class DAVG:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = Model().to(self.device)

    def train_new_model(self, data_file, label_file, batch_size=100, epochs=10, lr_rate=1e-3):
        """
        :param data_file: Numpy array containing mean word embeddings for every post
        :param label_file: Numpy array containing 1003 label vector for every post
        """

        l1_loss = torch.nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr_rate)

        for epoch in range(epochs):
            epoch_train_loss = 0.0
            start = time.time()
            file_reads = 0.
            with open(data_file, 'rb') as data, open(label_file, 'rb') as labels:
                try:
                    while True:
                        train_X = pickle.load(data)
                        train_y = pickle.load(labels)
                        data_loss = 0.0
                        file_reads += 1.

                        assert train_X.shape[0] == train_y.shape[0]
                        N = train_X.shape[0]

                        num_iterations = int(N / batch_size)

                        for iter in range(num_iterations):
                            batch_idxs = np.random.choice(a=range(N), size=batch_size)
                            batch_X = train_X[batch_idxs]
                            batch_y = train_y[batch_idxs]

                            # Converting to tensor
                            batch_X = torch.from_numpy(batch_X).to(self.device)
                            batch_y = torch.from_numpy(batch_y.astype('float32')).to(self.device)

                            optimizer.zero_grad()

                            probs = self.model(batch_X)

                            loss = l1_loss(probs, batch_y)
                            data_loss += loss / batch_size

                            loss.backward()

                            optimizer.step()

                        data_loss /= num_iterations

                        epoch_train_loss += data_loss

                except EOFError:
                    print('Finished training on whole data..')
                    pass
            end = time.time() - start
            print('[Training] Epoch loss %f [Time: %f]' % (epoch_train_loss / file_reads, end))
            self.save_model()

    def train_model(self, data_file, embeddings_file, validation_split=0.2, batch_size=5, epochs=1,
                    learning_rate=1e-3):
        """
        :param data_file: csv file consisting of post body and correspoding labels
        :param embeddings_file: gensim model containing the word embeddings
        """
        transformed_dataset = PostData(csv_file=data_file,
                                       transform=transforms.Compose([ExtractEmbeddings(embeddings_file), ToTensor()]))

        dataset_size = len(transformed_dataset)
        idxs = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        np.random.shuffle(idxs)

        train_idxs, val_idxs = idxs[split:], idxs[:split]
        train_sampler = SubsetRandomSampler(train_idxs)
        val_sampler = SubsetRandomSampler(val_idxs)

        train_dataloader = DataLoader(transformed_dataset, batch_size=batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(transformed_dataset, batch_size=batch_size, sampler=val_sampler)

        l1_loss = torch.nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_train_loss = 0.0
            iters = 0
            start = time.time()
            for i, batch_data in enumerate(train_dataloader):
                batch_X, batch_y = batch_data['body'].to(self.device), batch_data['labels'].to(self.device)

                optimizer.zero_grad()
                probs = self.model(batch_X)

                # loss = mse_loss(probs, batch_y)
                loss = l1_loss(probs, batch_y)
                epoch_train_loss += loss / batch_size

                loss.backward()
                optimizer.step()

                iters += 1

                # print('Iteration [%d] loss: %f' % (i, loss))
            end = time.time() - start

            print('[Training] Epoch loss %f [Time: %f]' % (epoch_train_loss / iters, end))
            epoch_val_loss = 0.0
            with torch.no_grad():
                iters = 0
                start = time.time()
                for i, batch_data in enumerate(val_dataloader):
                    batch_X, batch_y = batch_data['body'].to(self.device), batch_data['labels'].to(self.device)
                    probs = self.model(batch_X)
                    # val_loss = mse_loss(probs, batch_y)
                    val_loss = l1_loss(probs, batch_y)

                    epoch_val_loss += val_loss / batch_size

                    iters += 1
                end = time.time() - start
                print('[Validation] Epoch loss %f  [Time: %f]' % (epoch_val_loss / iters, end))

            # if epoch % 10 == 0 and epoch != 0:
            self.save_model()

    def test_new_model(self, data_file, label_file, batch_size=100, threshold=0.5):
        """
        :param data_file: Numpy array containing mean word embeddings for every post
        :param label_file: Numpy array containing 1003 label vector corresponding to each post
        """
        flag = True
        file_reads = 0.

        acc = tp = tn = fp = fn = 0.

        start = time.time()
        with open(data_file, 'rb') as data, open(label_file, 'rb') as labels:
            try:
                while flag:
                    train_X = pickle.load(data)
                    train_y = pickle.load(labels)
                    file_reads += 1.

                    # if file_reads > 5:
                    #     flag = False

                    N = train_X.shape[0]
                    num_iterations = int(N / batch_size)
                    batch_idx = 0
                    data_true_positive = 0.
                    data_true_negative = 0.
                    data_false_positive = 0.
                    data_false_negative = 0.
                    data_accuracy = 0.

                    for iter in range(num_iterations):
                        batch_X = train_X[batch_idx:batch_idx + batch_size]
                        batch_y = train_y[batch_idx:batch_idx + batch_size]
                        batch_idx += batch_size

                        batch_X = torch.from_numpy(batch_X).to(self.device)

                        with torch.no_grad():
                            probs = self.model(batch_X)

                            probs = probs.data.cpu().numpy()
                            # print('probability values', probs[0, :])
                            y_preds = np.where(probs < threshold, 0, 1)
                            y_gt = batch_y

                            pos_idxs = (y_gt == 1)
                            neg_idxs = (y_gt == 0)

                            if np.sum(pos_idxs) == 0:
                                data_true_positive += 0.
                                data_false_negative += 0.
                                print('No positive labels in ground truth')
                            else:
                                data_true_positive += np.mean((y_gt[pos_idxs] == y_preds[pos_idxs]).astype(int))
                                data_false_negative += np.mean(
                                    np.logical_xor(y_gt[pos_idxs].astype(bool), y_preds[pos_idxs].astype(bool)))

                            data_true_negative += np.mean((y_gt[neg_idxs] == y_preds[neg_idxs]).astype(int))
                            data_false_positive += np.mean(
                                np.logical_xor(y_gt[neg_idxs].astype(bool), y_preds[neg_idxs].astype(bool)))

                            data_accuracy += np.mean((y_preds == y_gt).astype(int))

                    acc += data_accuracy / num_iterations
                    tp += data_true_positive / num_iterations
                    tn += data_true_negative / num_iterations
                    fp += data_false_positive / num_iterations
                    fn += data_false_negative / num_iterations

            except EOFError:
                print('Finished training on whole data..')
                pass
        end = time.time() - start
        print('Test Accuracy: ', acc / file_reads)
        print('True Positive Rate: ', tp / file_reads)
        print('True Negative Rate: ', tn / file_reads)
        print('False Positive Rate: ', fp / file_reads)
        print('False Negative Rate: ', fn / file_reads)
        print('Time to predict: ', end)

    def test_model(self, test_data_file, embeddings_file, batch_size=5, threshold=0.5):
        """
        :param test_data_file: csv file consisting of post body and correspoding labels
        :param embeddings_file: gensim model containing the word embeddings
        """
        transformed_dataset = PostData(csv_file=test_data_file,
                                       transform=transforms.Compose([ExtractEmbeddings(embeddings_file), ToTensor()]))
        dataloader = DataLoader(transformed_dataset, batch_size=batch_size)
        with torch.no_grad():
            true_positive = 0.
            true_negative = 0.
            false_positive = 0.
            false_negative = 0.
            accuracy = 0.
            start = time.time()
            iters = 0
            for i, batch_data in enumerate(dataloader):
                batch_X, batch_y = batch_data['body'].to(self.device), batch_data['labels']

                probs = self.model(batch_X)
                probs = probs.data.cpu().numpy()

                y_preds = np.where(probs < threshold, 0, 1)
                y_gt = batch_y.numpy()

                # print('Probs')
                # print(probs[0, :])
                # print('Ground truth')
                # print(y_gt[0, :])

                pos_idxs = (y_gt == 1)
                neg_idxs = (y_gt == 0)

                if np.sum(pos_idxs) == 0:
                    true_positive += 0.
                    false_negative += 0.
                    print('No positive labels in ground truth')
                else:
                    true_positive += np.mean((y_gt[pos_idxs] == y_preds[pos_idxs]).astype(int))
                    false_negative += np.mean(np.logical_xor(y_gt[pos_idxs], y_preds[pos_idxs]).astype(int))

                true_negative += np.mean((y_gt[neg_idxs] == y_preds[neg_idxs]).astype(int))
                false_positive += np.mean(np.logical_xor(y_gt[neg_idxs], y_preds[neg_idxs]).astype(int))

                accuracy += np.mean((y_preds == y_gt).astype(int))
                iters += 1
        end = time.time() - start
        print('Test Accuracy: ', accuracy / iters)
        print('True Positive Rate: ', true_positive / iters)
        print('True Negative Rate: ', true_negative / iters)
        print('False Positive Rate: ', false_positive / iters)
        print('False Negative Rate: ', false_negative / iters)
        print('Time to predict: ', end)

    def save_model(self):
        torch.save(self.model.state_dict(), './checkpoints/model_params.pt')

    def load_model(self):
        self.model = Model().to(self.device)
        self.model.load_state_dict(torch.load('./checkpoints/model_params.pt'))
        self.model.eval()


if __name__ == '__main__':
    model = DAVG()
    model.load_model()
    # model.train_model(data_file='./data/baseline_train_data.csv', embeddings_file='./gensim_wordmodel_1_min',
    #                   batch_size=100, epochs=10)
    # model.train_new_model(data_file='./data/data1.pkl', label_file='./data/labels1.pkl', batch_size=100, epochs=1)
    model.test_new_model(data_file='./data/data1.pkl', label_file='./data/labels1.pkl', batch_size=100, threshold=0.5)
    # model.save_model()
