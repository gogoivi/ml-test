import numpy as np
from scipy.ndimage import label

def connected_comps(array, class_value):
    """
    Get connected components for a specific class.
    
    Args:
        array: Segmentation array with class labels
        class_value: Which class to find components for
    
    Returns:
        Labeled array where each connected component has a unique ID
    """
    binary_mask = (array == class_value).astype(np.uint8)
    labelled, num_components = label(binary_mask)
    return labelled

# FOR VESSELS: cutoff is 10
def removeSmallComponents(array, cutoff):
    newArray = np.zeros_like(array)
    
    for a in np.unique(array)[1:]:
        labelled_components = np.asarray(connected_comps(array, a), dtype="uint8")
        for n in np.unique(labelled_components)[1:]:
            xlocs, ylocs, zlocs = np.where(labelled_components == n)
            if xlocs.shape[-1] > cutoff:
                newArray[labelled_components==n] = a 

    return newArray


# FOR NERVE: cutoff a is 1
def keepOneComponent(array, a):
    newArray = np.zeros_like(array)
    labelled_components = np.asarray(connected_comps(array, a), dtype="uint8")
    print(f'Labelled nerve is {np.unique(labelled_components)}')
    maxVol = 0
    comp = -1
    
    for n in np.unique(labelled_components)[1:]:
        xlocs, ylocs, zlocs = np.where(labelled_components == n)
        if xlocs.shape[-1] > maxVol:
            maxVol = xlocs.shape[-1]
            comp = n
    for k in np.unique(array)[1:]:
        if k != a:
            newArray[array==k] = k
