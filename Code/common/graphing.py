import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_loss(loss1, loss2=None, title='Training vs Validation Loss',
                   label1='Training Loss', label2='Validation Loss', file_path=None):
    epochs = range(1, len(loss1) + 1)
    
    plt.figure(figsize=(8, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    plt.plot(epochs, loss1, label=label1, color='cyan')
    if loss2 is not None:
        plt.plot(epochs, loss2, label=label2, color='magenta')

    plt.xlabel('Epochs', color='white')
    plt.ylabel('Loss', color='white')
    plt.title(title, color='white')

    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color('white')

    plt.grid(True, color='gray', linestyle='--', alpha=0.5)
    ax.tick_params(colors='white')

    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    else:
        plt.show()


def plot_rewards(rewards1, rewards2=None, title='Total and Average Reward per Episode',
                      label1='Training Reward', label2='Testing Reward', file_path=None):
    
    epochs = range(1, len(rewards1) + 1)
    
    avg_rewards1 = [np.mean(rewards1[max(0, i-99):i+1]) for i in range(len(rewards1))]
    
    plt.figure(figsize=(10, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    plt.plot(epochs, rewards1, label=f'Total {label1}', color='cyan', marker='o', markersize=3)
    plt.plot(epochs, avg_rewards1, label=f'Average {label1}', color='orange', linewidth=2)

    if rewards2 is not None:
        avg_rewards2 = [np.mean(rewards2[max(0, i-99):i+1]) for i in range(len(rewards2))]
        plt.plot(epochs, rewards2, label=f'Total {label2}', color='magenta', markersize=3)
        plt.plot(epochs, avg_rewards2, label=f'Average {label2}', color='lime', linewidth=2)

    plt.xlabel('Epochs', color='white')
    plt.ylabel('Rewards', color='white')
    plt.title(title, color='white')

    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color('white')

    plt.grid(True, color='gray', linestyle='--', alpha=0.5)
    ax.tick_params(colors='white')

    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    else:
        plt.show()


def plot_steps(steps, title='Steps Taken per Episode', label='Steps taken', file_path=None):
    epochs = range(1, len(steps) + 1)

    plt.figure(figsize=(12, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    plt.plot(epochs, steps, label=label, color='cyan', marker='o', markersize=3)

    plt.xlabel('Episode', color='white')
    plt.ylabel('Steps', color='white')
    plt.title(title, color='white')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.grid(True, color='gray', linestyle='--', alpha=0.5)
    ax.tick_params(colors='white')
    
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    else:
        plt.show()



def plot_mean_std(mean, std, title, label, file_path=None):
    episodes = np.arange(len(mean))
    plt.figure(figsize=(12, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.plot(episodes, mean, label=label, color='cyan')
    plt.fill_between(episodes, mean - std, mean + std, alpha=0.3, color='magenta')
    plt.title(title, color='white')
    plt.xlabel("Episode", color='white')
    plt.ylabel(label, color='white')
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color('white')
    plt.grid(True, color='gray', linestyle='--', alpha=0.5)
    ax.tick_params(colors='white')
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()




def plot_comparison_distribution(tensor_list1, tensor_list2, label1='Real Noise', label2='Predicted Noise', title="Pixel Value Distribution Comparison", bins=50, file_path=None):
    """
    Plots and compares the histogram of pixel values from two datasets.
    Handles batched tensors.
    """
    if not isinstance(tensor_list1, list) or not isinstance(tensor_list2, list):
        raise TypeError("Both inputs must be lists of PyTorch tensors")
    if not all(isinstance(t, torch.Tensor) for t in tensor_list1) or not all(isinstance(t, torch.Tensor) for t in tensor_list2):
        raise TypeError("Each element in the lists must be a PyTorch tensor")
    
    # Flatten all pixel values
    all_pixel_values1 = np.concatenate([t.detach().cpu().numpy().flatten() for t in tensor_list1])
    all_pixel_values2 = np.concatenate([t.detach().cpu().numpy().flatten() for t in tensor_list2])
    
    # Set dark mode style
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8,6))
    
    # Plot histograms
    ax.hist(all_pixel_values1, bins=bins, alpha=0.75, label=label1, color='cyan', edgecolor='white')
    ax.hist(all_pixel_values2, bins=bins, alpha=0.75, label=label2, color='magenta', edgecolor='white')
    
    # Labels, title, legend
    ax.set_xlabel("Pixel Intensity", color='white')
    ax.set_ylabel("Frequency", color='white')
    ax.set_title(title, color='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    
    # Grid lines
    ax.grid(True, color='gray', linestyle='--', alpha=0.3)

    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    