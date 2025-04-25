import matplotlib.pyplot as plt

def plot_history(hist, fig_path='model_history.png'):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 5))  # Adjust figure size if needed

    plt.subplot(1, 2, 1)  # Create subplot for accuracy
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)  # Create subplot for loss
    plt.plot(hist.history['loss'][1:]) # Exclude first epoch because of ridiculously high loss
    plt.plot(hist.history['val_loss'][1:])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()  # Adjust layout for better spacing
    # plt.show()
    plt.savefig(fig_path)  # Save the figure
    plt.close()  