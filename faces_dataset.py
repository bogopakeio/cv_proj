"""Custom faces dataset."""
import os

import matplotlib.image as mpimg
import torch
from PIL import Image
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""

        label =-1
        if index < len(self.real_image_names):
            label = 0
            img_name = os.path.join(self.root_path,"real",self.real_image_names[index])
        else:
            label = 1
            img_name = os.path.join(self.root_path,"fake",self.fake_image_names[index - len(self.real_image_names)])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image,label
        #return torch.rand((3, 256, 256)), int(torch.randint(0, 2, size=(1, )))

    def __len__(self):
        """Return the number of images in the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""

        return len(self.real_image_names) + len(self.fake_image_names)
