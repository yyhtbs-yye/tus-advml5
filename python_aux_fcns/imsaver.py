import matplotlib.pyplot as plt
import numpy as np

def save_grid_images(images, labels, predicted, name_dict, nrow=8, show_only_incorrect=False):
    if show_only_incorrect:
        incorrect_indices = [i for i, (l, p) in enumerate(zip(labels, predicted)) if l != p]
        images = np.array([images[i] for i in incorrect_indices])
        labels = [labels[i] for i in incorrect_indices]
        predicted = [predicted[i] for i in incorrect_indices]
    
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))
    
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
    fig.subplots_adjust(hspace=0.5, wspace=0.1)
    
    for i, ax in enumerate(axes.flat):
        if i < n_images:
            ax.imshow(images[i])
            title = f'T:{name_dict[labels[i]]}/P:{name_dict[predicted[i]]}'
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("grid_figure.png")
