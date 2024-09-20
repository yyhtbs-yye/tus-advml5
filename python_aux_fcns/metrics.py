def calculate_accuracy(labels, predicted):

    # Calculate the total number of predictions
    total = labels.size(0)

    # Calculate the number of correct predictions
    correct = (predicted == labels).sum().item()

    # Calculate the accuracy
    accuracy = correct / total

    return accuracy

def calculate_confusion_matrix(labels, predicted):
    # Ensure binary output
    assert labels.max() <= 1 and labels.min() >= 0, "Labels must be binary."
    assert predicted.max() <= 1 and predicted.min() >= 0, "Predictions must be binary."
    
    # Calculate TP, TN, FP, FN
    TP = ((predicted == 1) & (labels == 1)).sum().item()
    TN = ((predicted == 0) & (labels == 0)).sum().item()
    FP = ((predicted == 1) & (labels == 0)).sum().item()
    FN = ((predicted == 0) & (labels == 1)).sum().item()
    
    return TP, TN, FP, FN

def calculate_precision(labels, predicted):
    TP, TN, FP, FN = calculate_confusion_matrix(labels, predicted)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    return precision

def calculate_specificity(labels, predicted):
    TP, TN, FP, FN = calculate_confusion_matrix(labels, predicted)
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    return specificity

def calculate_sensitivity(labels, predicted):
    TP, TN, FP, FN = calculate_confusion_matrix(labels, predicted)
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    return sensitivity

def calculate_f1_score():
    precision, sensitivity = calculate_precision(labels, predicted), calculate_sensitivity(labels, predicted)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
    return f1_score


