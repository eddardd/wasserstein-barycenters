import numpy as np
from skimage.transform import resize


def project_simplex(x):
    """Project Simplex

    Projects an arbitrary vector :math:`\mathbf{x}` into the probability simplex, such that,

    .. math:: \tilde{\mathbf{x}}_{i} = \dfrac{\mathbf{x}_{i}}{\sum_{j=1}^{n}\mathbf{x}_{j}}

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Numpy array of shape (n,)

    Returns
    -------
    y : :class:`numpy.ndarray`
        numpy array lying on the probability simplex of shape (n,)
    """
    x[x < 0] = 0
    if np.isclose(sum(x), 0):
        y = np.zeros_like(x)
    else:
        y = x.copy() / sum(x)
    return y


def create_digits_image(images, labels, digit=0, grid_size=64, n_digits=15, original_size=28, is_distribution=True):
    batch = images[np.where(labels==digit)[0]]
    images = []
    for i in range(n_digits):
        grid = np.zeros([grid_size, grid_size])
        final_size = np.random.randint(original_size // 2, original_size * 2) 
        grid_cells = np.arange(0, grid_size - final_size)
        img = resize(batch[np.random.randint(0, len(batch))], (final_size, final_size)) 
        p = [10] + [1] * (len(grid_cells) - 2) + [10]
        center_x = np.random.choice(grid_cells, size=1, p=np.array(p) / sum(p))[0] 
        center_y = np.random.choice(grid_cells, size=1, p=np.array(p) / sum(p))[0] 
        grid[center_x:center_x+final_size, center_y:center_y+final_size] = img.copy()
        if is_distribution:
            grid = grid / np.sum(grid)
        images.append(grid.reshape(1, grid_size, grid_size))
    images = np.array(images).reshape(-1, grid_size ** 2)

    return images