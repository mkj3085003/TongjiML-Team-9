from torch import optim
from tqdm import tqdm
import torch
import torch.nn as nn
def train(model, train_loader, val_loader, num_epoch, learning_rate,model_path, device, writer):
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    best_acc = 0.0

    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        for i, batch in enumerate(tqdm(train_loader)):
            # A batch consists of features data and corresponding labels.
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, train_pred = torch.max(outputs, 1)
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                _, val_pred = torch.max(outputs, 1)
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += loss.item()

        # Record the accuracy and lr.
        writer.add_scalar('Acc/train', train_acc / len(train_loader.dataset), epoch)
        writer.add_scalar('Acc/valid', val_acc / len(val_loader.dataset), epoch)
        writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        # Print the information.
        print(
            f'[{epoch + 1:03d}/{num_epoch:03d}] Train Acc: {train_acc / len(train_loader.dataset):3.5f} Loss: {train_loss / len(train_loader):3.5f} | Val Acc: {val_acc / len(val_loader.dataset):3.5f} loss: {val_loss / len(val_loader):3.5f}')

        # If the model improves, save a checkpoint at this epoch
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'saving model with acc {best_acc / len(val_loader.dataset):.5f}')

        print(f"{epoch + 1} lr: {optimizer.state_dict()['param_groups'][0]['lr']}")


    print(f'saving model with acc {best_acc / len(val_loader.dataset):.5f}')


# 使用示例
# train(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epoch, model_path, device, writer)
