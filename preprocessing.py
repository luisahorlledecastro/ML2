import kagglehub
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def import_data():
    # import data
    # Download latest version
    path = kagglehub.dataset_download("anaghachoudhari/pcos-detection-using-ultrasound-images")

    # prepare pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # make sure images have the same size
        transforms.ToTensor()
    ])

    # load training data. -> placeholder for now
    train_dataset = datasets.ImageFolder(root=os.path.join(path, "data", "train"), transform=transform)

    # load test data. -> placeholder for now
    test_dataset = datasets.ImageFolder(root=os.path.join(path, "data", "test"), transform=transform)

    return train_dataset, test_dataset

train_dataset, test_dataset = import_data()


def display_examples(data, show=False):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    infected_example, _ = data[0]
    not_infected_example, _ = data[-1]

    ax[0].imshow(infected_example.permute(1, 2, 0))
    ax[0].set_title("Infected")

    ax[1].imshow(not_infected_example.permute(1, 2, 0))
    ax[1].set_title("Not Infected")

    if show:
        plt.show()


if __name__ == "__main__":
    pass
