import numpy as np
import torch
from ml_classification.model.LogisticRegressionClassifier.logisticRegression import LogisticRegression

class OneVsAllLRClassifier:
    def __init__(self, input_dim, output_dim=41, random_state=None, num_epoch=50, learning_rate=1e-2):
        self.input_dim = input_dim
        self.num_classes = output_dim
        self.classifiers = {}
        self.random_state = random_state
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate

    def train(self, train_data, train_labels, device):
        for i in range(self.num_classes):
            # Train a binary classifier for class i vs. all other classes
            cond = train_labels == i
            train_data_i = train_data.to(device)
            train_labels_i = torch.tensor(cond, dtype=torch.float).to(device)

            model = LogisticRegression(self.input_dim).to(device)
            criterion = torch.nn.BCELoss(reduction='sum')
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

            for epoch in range(self.num_epoch):
                y_pred = model(train_data_i)
                loss = criterion(y_pred, train_labels_i.view(-1, 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.classifiers[str(i)] = model
            print(f"Classifier for class {i} trained.")

    def predict(self, test_data, device):
        results = torch.zeros(len(test_data), self.num_classes, dtype=torch.float).to(device)
        for i in range(self.num_classes):
            classifier = self.classifiers[str(i)]
            predictions = classifier(test_data)
            results[:, i] = predictions.squeeze()
        return results

    def predict_and_write_to_file(self, test_loader, device, output_file='prediction.csv'):
        self.predictions = torch.tensor([]).to(device)

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                features = batch.to(device)
                pred = self.predict(features, device)

                self.predictions = torch.cat((self.predictions, pred), dim=0)
                print(f"Batch {i + 1} processed.")

        with open(output_file, 'w') as f:
            f.write('Id,Class\n')
            for i, y in enumerate(self.predictions):
                class_id = torch.argmax(y).item()
                f.write(f'{i},{class_id}\n')

