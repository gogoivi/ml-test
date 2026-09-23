# https://www.sciencedirect.com/science/article/pii/S1746809426000662#cited-by
#  Changed to 3d instead of 2d and attempting to compute per class

from scipy.ndimage import binary_erosion
import gudhi
import numpy as np
from torch import nn
import torch

def get_surface_points(binary_mask):
    """
    binary_mask: numpy array of shape (D, H, W) with values 0 or 1
    returns: numpy array of shape (N, 3) containing [z, y, x] coordinates of surface voxels
    """
    eroded = binary_erosion(binary_mask)  # Shrinks by 1 voxel
    surface = binary_mask & ~eroded  # Original minus eroded = surface only
    coords=np.argwhere(surface)
    return coords

def compute_persistence_diagram(points, max_dimension=1,size:int=20):
    """
    points: numpy array of shape (N, 3) - your surface coordinates
    max_dimension: 0 for components only, 1 for components + holes
    returns: persistence diagram as list of (dimension, (birth, death)) tuples
    """
    # Create Rips complex from point cloud
    rips = gudhi.RipsComplex(points=points, max_edge_length=20)
    
    # Build simplex tree up to desired dimension
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension + 1)
    
    # Compute persistence
    simplex_tree.compute_persistence()
    
    return simplex_tree.persistence()

def subsample_points(points, max_points=1000):
    """
    points: (N, 3) array of surface coordinates
    max_points: maximum number of points to keep
    """
    if len(points) <= max_points:
        return points
    
    # Randomly select max_points indices
    indices = np.random.choice(len(points), max_points, replace=False)
    return points[indices]

def compute_wasserstein_distance(X:list[tuple[int, tuple[float, float]]],Y:list[tuple[int, tuple[float, float]]]):
    dim0_X=np.array([item[1] for item in X if item[0]==0 and item[1][1] != np.inf])
    dim1_X=np.array([item[1] for item in X if item[0]==1 and item[1][1] != np.inf])
    dim0_Y=np.array([item[1] for item in Y if item[0]==0 and item[1][1] != np.inf])
    dim1_Y=np.array([item[1] for item in Y if item[0]==1 and item[1][1] != np.inf])
    dim0_dist=gudhi.wasserstein.wasserstein_distance(dim0_X,dim0_Y)
    dim1_dist=gudhi.wasserstein.wasserstein_distance(dim1_X,dim1_Y)
    return dim0_dist, dim1_dist

def get_class_mask(pred_logits, class_idx):
    """
    pred_logits: (C, D, H, W) tensor - single sample
    class_idx: which class (1=nerve, 2=vessel)
    returns: (D, H, W) numpy array of 0s and 1s
    """
    pred_classes = torch.argmax(pred_logits, dim=0)  # (D, H, W)
    binary_mask = (pred_classes == class_idx)        # Boolean tensor
    return binary_mask.cpu().numpy().astype(np.uint8)

def compute_total_persistence(persistence_diagram,MIN_PEN:float=1.0):
    """
    Given a persistence diagram (list of (dim, (birth, death)) tuples),
    compute sum of all finite bar lengths.
    Returns: d0 persistence, d1 persistence
    """
    d0_persistence=np.array([item[1][1] - item[1][0] for item in persistence_diagram if item[0]==0 and item[1][1] != np.inf])
    d1_persistence=np.array([item[1][1] - item[1][0] for item in persistence_diagram if item[0]==1 and item[1][1] != np.inf])
    d0=np.sum(d0_persistence)
    d1=np.sum(d1_persistence)
    if d1<MIN_PEN:
        d1=MIN_PEN
    if d0<MIN_PEN:
        d0=MIN_PEN

    return d0,d1

def get_class_mask_from_target(target, class_idx):
    """
    target: (D, H, W) tensor of class labels
    """
    binary_mask = (target == class_idx)
    return binary_mask.cpu().numpy().astype(np.uint8)

class TopologyAwareLoss(nn.Module):
    def __init__(self, alpha, beta, warmup_epochs:int=25):
        super().__init__()
        """Assuming class 1 is background iteratinfg through 1 and 2"""
        self.alpha=alpha
        self.beta=beta
        self.warmup_epochs=warmup_epochs
        
    def forward(self, pred, target, current_epoch,N:int=1):
        """N is amount of epochs that it skips before doing these computations"""
        d0_total=0
        d1_total=0
        d0_class1=0
        d1_class1=0
        d0_class2=0
        d1_class2=0
        if (current_epoch<=self.warmup_epochs) or (current_epoch%N !=0):
            return 1
        else:
            for i in range(pred.shape[0]):
                pred_points_class1 = subsample_points(get_surface_points(get_class_mask(pred[i],1)))
                target_points_class1 = subsample_points(get_surface_points(get_class_mask_from_target(target[i],1)))
                pred_points_class2 = subsample_points(get_surface_points(get_class_mask(pred[i],2)))
                target_points_class2 = subsample_points(get_surface_points(get_class_mask_from_target(target[i],2)))

                # If nothing is predicted and nothing exists for class1 (got it right)
                if len(pred_points_class1) == 0 and len(target_points_class1) == 0: 
                    d0_class1+=0
                    d1_class1+=0
                # If nothing is predicted or nothing exists for class1 (bad - predicted when nothing was there or didnt predict when something was there)
                elif len(pred_points_class1) == 0 or len(target_points_class1) == 0:
                    if len(pred_points_class1) == 0:
                        target_class1_PD=compute_persistence_diagram(target_points_class1)
                        d0,d1=compute_total_persistence(target_class1_PD)
                    else: 
                        pred_class1_PD=compute_persistence_diagram(pred_points_class1)
                        d0,d1=compute_total_persistence(pred_class1_PD)    
                    d0_class1+=d0
                    d1_class1+=d1

                else:
                    pred_class1_PD=compute_persistence_diagram(pred_points_class1)
                    target_class1_PD=compute_persistence_diagram(target_points_class1)
                    d0_class1,d1_class1=compute_wasserstein_distance(pred_class1_PD,target_class1_PD)

                if len(pred_points_class2) == 0 and len(target_points_class2) == 0: 
                    d0_class2+=0
                    d1_class2+=0
                elif len(pred_points_class2) == 0 or len(target_points_class2) == 0:
                    if len(pred_points_class2) == 0:
                        target_class2_PD=compute_persistence_diagram(target_points_class2)
                        d0,d1=compute_total_persistence(target_class2_PD)
                    else: 
                        pred_class2_PD=compute_persistence_diagram(pred_points_class2)
                        d0,d1=compute_total_persistence(pred_class2_PD)
                    d0_class2+=d0
                    d1_class2+=d1
                else:
                    # Nothing empty so compute wasserstein distance
                    pred_class2_PD=compute_persistence_diagram(pred_points_class2)
                    target_class2_PD=compute_persistence_diagram(target_points_class2)
                    d0_class2,d1_class2=compute_wasserstein_distance(pred_class2_PD,target_class2_PD)

                d0_total+=(d0_class1+d0_class2)/2
                d1_total+=(d1_class1+d1_class2)/2

            return 1+self.alpha*(d0_total/pred.shape[0])+self.beta*(d1_total/pred.shape[0])