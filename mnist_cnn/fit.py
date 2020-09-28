import torch


def run_epoch(dl, model, loss_func, opt=None):
    total_loss = 0
    total_size = 0
    cnt_true = 0
    for batch in dl:
        x, y = batch['x'], batch['y']
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        total_loss += loss * x.size()[0]
        total_size += x.size()[0]
        cnt_true += torch.sum(torch.argmax(y_pred, dim=1) == y).item()
        if opt:
            loss.backward()
            opt.step()
            opt.zero_grad()

    return total_loss / total_size, cnt_true / total_size


def fit(train_dl, valid_dl, model, loss_func, opt, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss, _ = run_epoch(train_dl, model, loss_func, opt)
        
        model.eval()
        with torch.no_grad():
            valid_loss, accuracy = run_epoch(valid_dl, model, loss_func)
            
        print(f'Epoch {epoch} -  training loss: {train_loss}   validation loss: {valid_loss}   accuracy: {accuracy}')
