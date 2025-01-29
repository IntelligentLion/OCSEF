import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ Split image into patches and then embed them.
    Parameters
    ----------
    img_size : int
        Size of the image (it is a square).

    patch_size : int
        Size of the patch (it is a square).

    in_channels : int
        Number of input channels.

    embed_dim : int
        Dimension of the embedding.

    Attributes
    ----------
    n_patches : int
        Number of patches inside the image.
    proj : nn.Conv2d
        Convolutional layer that does both the splitting and the embedding.
    """
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_channels, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """
        x = self.proj(x)