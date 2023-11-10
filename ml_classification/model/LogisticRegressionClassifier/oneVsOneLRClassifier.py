import torch
from collections import Counter
from itertools import combinations
from ml_classification.model.LogisticRegressionClassifier.logisticRegression import  LogisticRegression

class OneVsOneLRClassifier:
    def __init__(self, input_dim, output_dim=41,random_state=None,num_epoch=50,learning_rate=1e-2):
        self.input_dim = input_dim
        self.num_classes = output_dim
        self.classifiers = {}
        self.random_state = random_state
        self.num_epoch=num_epoch
        self.learning_rate=learning_rate

    def train(self, train_data, train_labels, device):
        for i, j in combinations(range(self.num_classes), 2):
            cond = torch.logical_or(train_labels == i, train_labels == j)
            train_data_ij = train_data[cond].to(device)
            train_labels_ij = train_labels[cond].to(device)  # train_label
            model = LogisticRegression(self.input_dim).to(device)
            criterion = torch.nn.BCELoss(reduction='sum')
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

            for epoch in range(self.num_epoch):
                y_pred = model(train_data_ij)
                loss = criterion(y_pred, train_labels_ij.float().view(-1, 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.classifiers[f'{i}_{j}'] = model
            print(f"Classifier for classes {i} and {j} trained.")


    #每个 predictions 都是针对一对类别的预测概率，而 results 列表包含了所有这些预测结果。
    def predict(self, test_data):
        results = []
        for i, j in combinations(range(self.num_classes), 2):
            classifier = self.classifiers[f'{i}_{j}']
            predictions = classifier(test_data)
            results.append(predictions)
        return results

    #确定最终类别的标签
    def get_final_predictions(self,results):
        final_predictions = []
        for i in range(len(results[0])):
            class_votes = Counter()
            for prediction in results:
                class_votes[prediction[i].argmax().item()] += 1
            final_predictions.append(class_votes.most_common(1)[0][0])

        return torch.tensor(final_predictions)

    def predict_and_write_to_file(self, test_loader, device, output_file='prediction.csv'):
        self.predictions = torch.tensor([])

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                features = batch.to(device)
                pred = self.get_final_predictions(self.predict(features))
                self.predictions = torch.cat((self.predictions, pred), dim=0)
                print(f"Batch {i + 1} processed.")

        with open(output_file, 'w') as f:
            f.write('Id,Class\n')
            for i, y in enumerate(self.predictions):
                f.write(f'{i},{y.item()}\n')