import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import seaborn as sns

def get_accuracy(preds, targets):
        assert preds.shape[0] == targets.shape[0]
        correct = torch.eq(preds.argmax(dim=1), targets).float().sum().item()
        accuracy = correct/preds.shape[0]
        return accuracy

def binary_accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct

def get_next_batch(loaders, iters, src, device):
    inputs, labels = None, None

    try:
        inputs, labels = next(iters[src])
    except StopIteration:
        iters[src] = iter(loaders[src])
        inputs, labels = next(iters[src])

    return inputs.to(device), labels.to(device)
    
def visualize_combined_tsne(features, labels, val_features, val_labels, save_dir, file_name, name):
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate=200.0)
    combined_features = np.concatenate([features, val_features], axis=0)
    transformed_features = tsne.fit_transform(combined_features)

    # Split the transformed features back
    features_0_transformed = transformed_features[:len(features)]
    val_transformed = transformed_features[len(features):]

    # Define marker styles and colors
    marker_styles = {'source': 'o', 'target': '^'}  # circle for source, cross for target
    colors = sns.color_palette("Set2", 4)  # Use a Seaborn color palette
    label_names = ['Normal', 'Inner', 'Outer', 'ball']  # Label names for the legend

    # Create a figure for the t-SNE plot
    plt.figure(figsize=(12, 10))

    # Visualize t-SNE for source set
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(features_0_transformed[idx, 0], features_0_transformed[idx, 1], 
                    label=f'Source: {label_names[label]}', marker=marker_styles['source'], 
                    color=colors[label], s=500)

    # Visualize t-SNE for target set
    for label in np.unique(val_labels):
        idx = val_labels == label
        plt.scatter(val_transformed[idx, 0], val_transformed[idx, 1], 
                    label=f'Target: {label_names[label]}', marker=marker_styles['target'], 
                    color=colors[label], s=500)
    
    # Remove axis ticks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    save_path = os.path.join(save_dir, file_name)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()






def visualize_tsne_and_confusion_matrix(features, all_labels, cm, save_dir, file_name):
    """
    Visualize t-SNE and confusion matrix side-by-side with percentages.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.manifold import TSNE
    import os

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate=200.0)
    transformed_features = tsne.fit_transform(features)

    # Label names
    label_names = ['Normal','Ball' ,'Inner', 'Outer']

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Visualize t-SNE with legend
    for label, name in enumerate(label_names):
        ax1.scatter(
            transformed_features[np.array(all_labels) == label, 0], 
            transformed_features[np.array(all_labels) == label, 1], 
            label=name, s=50
        )
    ax1.set_xticks([])  # Remove x-axis ticks
    ax1.set_yticks([])  # Remove y-axis ticks
    ax1.legend(fontsize=15, title="Classes")
    ax1.set_title("t-SNE Visualization", fontsize=20)

    # Convert confusion matrix to percentage
    cm_percentage = cm  # Scale to percentage
    annot = np.array([
        ["{:.1f}%".format(value) if value > 0 else "" for value in row] 
        for row in cm_percentage
    ])

    # Visualize confusion matrix
    sns.heatmap(
        cm_percentage, annot=annot, cmap='Blues', fmt='', 
        xticklabels=label_names, yticklabels=label_names, ax=ax2,
        annot_kws={"size": 15}  # Annotation (numbers) font size
    )
    #ax2.set_title('Confusion Matrix (%)', fontsize=20)
    ax2.set_xlabel('Predicted Label', fontsize=20)
    ax2.set_ylabel('True Label', fontsize=20)

    # Set font size for axis tick labels
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)

    # Save the combined plot
    save_path = os.path.join(save_dir, file_name)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



