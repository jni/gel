import heapq

import numpy as np
import scipy.ndimage as nd
import scipy.spatial.distance as distance

def gel(image, num_superpixels, use_channels=False, num_iter=20, 
        mode='viscosity'):
    """Find GEL superpixels in an image.

    Parameters
    ----------
    image : np.ndarray (arbitrary dimension)
        The image in which to find the superpixels
    num_superpixels : int
        The desired number of superpixels
    use_channels : bool (optional)
        Whether to treat the last dimension of `image` as the channels
        (default: False)
    num_iter : int (optional)
        The number of geodesic expansion and recentering iterations to run
        (default: 20)
    mode : string (optional)
        Whether to treat the image as a viscosity map (default).

    Returns
    -------
    superpixels : np.ndarray (same shape as `image`)
        The superpixels found by GEL.
    """
    ndim = float(image.ndim)
    spacing = int(np.floor((image.size / num_superpixels) ** (1 / ndim)))
    slices = [slice(spacing/2, None, spacing)] * image.ndim
    centers = np.zeros(image.shape, np.uint8)
    centers[slices] = 1
    centers = nd.label(centers)[0]
    centers_old = None
    converged = False
    uniques = range(1, centers.max() + 1)
    for i in range(num_iter):
        if converged:
            break
        superpixels = geodesic_expansion(centers, image, mode)
        centers_new = label_centers_of_mass(superpixels, uniques)
        if i > 0 and centers_old == centers_new:
            break
        centers_old = centers_new
        centers = volume_of_labels(centers_new)
    return superpixels


def volume_of_labels(centers, shape):
    """Return a volume in which the given coordinates are point labels.

    Parameters
    ----------
    centers : list of tuples
        A list of coordinates (possibly fractional).
    shape : tuple
        The shape of the volume containing the centers.

    Returns
    -------
    volume : np.ndarray (shape given by `shape`)
    """
    volume = np.zeros(shape, int)
    for i, center in enumerate(centers):
        center = [int(np.floor(coord)) for coord in center]
        volume[center] = i+1
    return volume

def label_centers_of_mass(labels, uniques=None):
    """Find the center of mass of each label assuming uniform weights.

    Parameters
    ----------
    labels : np.ndarray, integer type
        A label field.
    uniques : list of int (optional)
        The labels for which to compute the centers of mass. 

    Returns
    -------
    centers : list of tuples
        Each tuple is a set of coordinates (of length labels.ndim) for the 
        center of mass of the corresponding label.

    Notes
    -----
    This function will be slow if called repeatedly without a `uniques` input,
    as it needs to run `np.unique` on the input array. 
    """
    if uniques is None:
        uniques = np.unique(labels)
        if uniques[0] == 0:
            uniques = uniques[1:]
    centers = nd.measurements.center_of_mass(labels, labels, uniques)
    return centers

def neighbors(coords, shape, connectivity=1):
    """Return the neighbors of a set of coordinates.

    Parameters
    ----------
    coords : array-like
        The coordinates for which we want neighbor coordinates.
    shape : array-like
        The shape of the array including coords. Used to check for borders.
    connectivity : int (optional), min=1, max=len(shape)
        The connectivity of the neighborhood.

    Returns
    -------
    neighbors : np.ndarray (num_neighbors x len(shape))
        The coordinates of neighboring array elements.
    """
    coords = np.atleast_2d(coords)
    shape = np.asarray(shape)
    ndim = len(shape)
    n_elem = 3 ** ndim
    footprint = nd.generate_binary_structure(ndim, connectivity)
    footprint.ravel()[n_elem / 2] = 0
    neighbors = coords + (np.asarray(footprint.nonzero()).T - np.ones(ndim))
    not_border = True - ((neighbors < 0).any(axis=1) + 
                         (neighbors >= shape).any(axis=1))
    return neighbors[not_border].astype(int)

def geodesic_expansion(labels, image, mode='viscosity', connectivity=1):
    """Expand the location of labels into a geodesic space defined by image.

    Parameters
    ----------
    labels : numpy array (integer type)
        The initial location of the labels (typically sparse)
    image : numpy array, dimensions = labels.ndim or labels.ndim + 1
        The space along which labels must expand. It can either be
        single-channel (same dimension as `labels`) or multi-channel (one more
        dimension than `labels`; the last dimension is assumed to be the
        channels).
    mode : string (optional)
        Whether to treat the image values as a `viscosity` (default) or a 
        `feature`. In the first case, the cost of expanding from pixel `x` to
        pixel `y` is `image[y]`. In the second, it is `d(image[x], image[y])`.
    connectivity : int (optional)
        The connectivity defining neighboring pixels. It is an int between 1
        and `labels.ndim`, inclusive. (default: 1)

    Returns
    -------
    labels : numpy array (integer type)
        The resulting label field
    """
    label_locations = labels.nonzero()
    initial_labels = labels[label_locations]
    distance_heap = [(0, coord, label) for coord, label in 
         zip(np.transpose(label_locations), initial_labels)]
    if mode == 'viscosity':
        def dist(img, src, dst): return img[dst]
    else:
        def dist(img, src, dst): return distance.euclidean(img[src], img[dst])
    labels_out = np.zeros_like(labels)
    while len(distance_heap) > 0:
        nearest = heapq.heappop(distance_heap)
        d, loc, lab = nearest
        if labels_out[loc] == 0:
            labels_out[loc] = lab
            for n in neighbors(loc, labels_out.shape, connectivity):
                if labels_out[n] == 0:
                    next_d = d + dist(image, loc, n)
                    heapq.heappush(distance_heap, (next_d, n, lab))
    return labels_out
