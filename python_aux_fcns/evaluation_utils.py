import torch 

def get_all(model, dataloader, to_cpu=True):
    model.eval()  # Set the model to evaluation mode
    Xs, ys, y_hats = [], [], []  # Initialize empty lists to accumulate batches

    with torch.no_grad():
        for X, y in dataloader:
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
            
            y_hat = model(X)
            _, predicted = torch.max(y_hat.data, 1) # [0: B, 1: C]
            
            if to_cpu:
                X = X.cpu()
                y = y.cpu()
                predicted = predicted.cpu()
            
            # Accumulate the batch results
            Xs.append(X)
            ys.append(y)
            y_hats.append(predicted)

    # Concatenate all accumulated batches
    Xs = torch.cat(Xs, dim=0) # [Xs, Xs, Xs] Xs.shape=[B, C], [[B, C], ...[B, C]] -> [K*B, C]
    ys = torch.cat(ys, dim=0)
    y_hats = torch.cat(y_hats, dim=0)
    
    return Xs, ys, y_hats

def unnormalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # The mean and std have to be reshaped to [C, 1, 1] to match the tensor dimensions.
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    
    if images.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    images.mul_(std).add_(mean)  # This modifies the tensor in-place
    return images

def get_imbalanced_dataset_weight(dataset):
    class_counts = {}
    for it in dataset:
        _, label = it  # Assuming your dataset has a 'targets' attribute
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    # Calculate weight for each sample in the training set
    weights = []
    for it in dataset:
        _, label = it
        weights.append(1 / class_counts[label])
    
    return weights
