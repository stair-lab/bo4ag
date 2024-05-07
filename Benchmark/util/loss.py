import torch

def calcMSE(model, train_X, lookup, **kwargs):
    seed = kwargs.get("seed", None)
    m = kwargs.get("m", None)  # size of validation set
    device = kwargs.get("device", None)

    torch.manual_seed(seed + 1234)

    val_X = torch.rand(m, 2, dtype=torch.float64, device=device) * 2150
    val_Y = model.posterior(val_X).mean.detach()
    train_Y = model.posterior(train_X).mean.detach()

    train_loss = torch.mean((train_Y - lookup(train_X)) ** 2)
    val_loss = torch.mean((val_Y - lookup(val_X)) ** 2)
    return train_loss, val_loss


def calcR2(model, train_X, lookup, **kwargs):
    seed = kwargs.get("seed", None)
    m = kwargs.get("m", None)  # size of validation set
    device = kwargs.get("device", None)

    torch.manual_seed(seed + 1234)

    val_X = torch.rand(m, 2, dtype=torch.float64, device=device) * 2150
    val_Y = model.posterior(val_X).mean.detach()
    train_Y = model.posterior(train_X).mean.detach()

    # Calculate the mean of the training labels
    y_train_mean = lookup(train_X).mean()
    y_val_mean = lookup(val_X).mean()

    # Calculate total sum of squares (TSS)
    TSS_train = torch.sum((lookup(train_X) - y_train_mean) ** 2)
    TSS_val = torch.sum((lookup(val_X) - y_val_mean) ** 2)

    # Calculate residual sum of squares (RSS)
    RSS_train = torch.sum((train_Y - lookup(train_X)) ** 2)
    RSS_val = torch.sum((val_Y - lookup(val_X)) ** 2)

    # Calculate R^2
    R2_train = 1 - (RSS_train / TSS_train)
    R2_val = 1 - (RSS_val / TSS_val)

    return R2_train, R2_val