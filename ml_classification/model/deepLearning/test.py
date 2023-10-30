import numpy as np
import torch

def test(model, test_loader, device, output_file='prediction.csv'):
    model.eval()
    pred = np.array([], dtype=np.int32)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            features = batch
            features = features.to(device)
            outputs = model(features)

            _, test_pred = torch.max(outputs, 1)
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

    with open(output_file, 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))