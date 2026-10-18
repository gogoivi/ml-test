# https://www.sciencedirect.com/science/article/pii/S1746809426000662#cited-by
#  Changed to 3d instead of 2d and attempting to compute per class

from scipy.ndimage import binary_erosion
import gudhi
import numpy as np
from torch import nn
import torch
from gudhi.wasserstein import wasserstein_distance
from gudhi.sklearn import RipsPersistence
from joblib import Parallel, delayed
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import linear_sum_assignment
import numpy as np

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

def subsample_points(points, max_points=200):
    """
    points: (N, 3) array of surface coordinates
    max_points: maximum number of points to keep
    """
    if len(points) <= max_points:
        return points
    
    # Randomly select max_points indices
    indices = np.random.choice(len(points), max_points, replace=False)
    return points[indices]

def compute_wasserstein_distance(X,Y):
    # dim0_X=np.array([item[1] for item in X if item[0]==0 and item[1][1] != np.inf])
    dim0_X=X[0]
    dim0_X=dim0_X[dim0_X[:,1] != np.inf]
    # dim1_X=np.array([item[1] for item in X if item[0]==1 and item[1][1] != np.inf])
    # dim0_Y=np.array([item[1] for item in Y if item[0]==0 and item[1][1] != np.inf])
    # dim1_Y=np.array([item[1] for item in Y if item[0]==1 and item[1][1] != np.inf])
    dim1_X=X[1]
    dim1_X=dim1_X[dim1_X[:,1] != np.inf]
    dim0_Y=Y[0]
    dim0_Y=dim0_Y[dim0_Y[:,1] != np.inf]
    dim1_Y=Y[1]
    dim1_Y=dim1_Y[dim1_Y[:,1] != np.inf]
    if len(dim0_X) == 0:
        dim0_X = np.empty((0, 2))
    if len(dim1_X) == 0:
        dim1_X = np.empty((0, 2))
    if len(dim0_Y) == 0:
        dim0_Y = np.empty((0, 2))
    if len(dim1_Y) == 0:
        dim1_Y = np.empty((0, 2))
    dim0_dist=wasserstein_distance(dim0_X,dim0_Y)
    dim1_dist=wasserstein_distance(dim1_X,dim1_Y)
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
    d0_filtered=persistence_diagram[0][persistence_diagram[0][:,1]!= np.inf]
    d1_filtered=persistence_diagram[1][persistence_diagram[1][:,1]!= np.inf]
    d0_persistence = d0_filtered[:, 1] - d0_filtered[:, 0]
    d1_persistence = d1_filtered[:, 1] - d1_filtered[:, 0]
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
    def __init__(self, alpha, beta, warmup_epochs:int=25,classes_wo_background:int=2):
        super().__init__()
        """Assuming class 1 is background iteratinfg through 1 and 2, this is the non-differentiable implementation"""
        self.alpha=alpha
        self.beta=beta
        self.warmup_epochs=warmup_epochs
        self.classes=classes_wo_background
        
    def forward(self, pred, target, current_epoch,N:int=1):

        d0_total=0
        d1_total=0

        if (current_epoch<=self.warmup_epochs) or (current_epoch%N !=0):
            return 1
        
        # # Parallel processing of all samples
        # results = Parallel(n_jobs=4)(
        #     delayed(process_single_sample)(pred[i], target[i])
        #     for i in range(pred.shape[0])
        # )
        
        # # Aggregate results
        # d0_total = sum((r[0] + r[2]) / 2 for r in results)
        # d1_total = sum((r[1] + r[3]) / 2 for r in results)

        for j in range(self.classes):
            surfaces = []
            metadata = []  # Track: (sample_idx, is_pred, is_empty)

            for i in range(pred.shape[0]):
                pred_surface = get_surface_points(get_class_mask(pred[i], j+1))
                target_surface = get_surface_points(get_class_mask_from_target(target[i], j+1))

                if len(pred_surface)!=0:
                    surfaces.append(subsample_points(pred_surface, max_points=200))
                    metadata.append((i,1,0))
                else:
                    metadata.append((i,1,1))

                if len(target_surface) !=0:
                    surfaces.append(subsample_points(target_surface, max_points=200))
                    metadata.append((i,0,0))
                else:
                    metadata.append((i,0,1))

            if len(surfaces) == 0:
                continue  # Skip to next class, add nothing to d0_total/d1_total

            metadata_non_empty=[items for items in metadata if items[2]==0]
            rp = RipsPersistence(homology_dimensions=[0, 1], threshold=20, n_jobs=-1)
            all_diagrams = rp.fit_transform(surfaces)

            # Create lookup arrays - one slot per sample
            pred_diagrams = [None] * pred.shape[0]
            target_diagrams = [None] * pred.shape[0]

            # Fill in using metadata_non_empty (which aligns with all_diagrams)
            for k, (sample_idx, is_pred, is_empty) in enumerate(metadata_non_empty):
                if is_pred == 1:
                    pred_diagrams[sample_idx] = all_diagrams[k]
                else:
                    target_diagrams[sample_idx] = all_diagrams[k]

            for i in range(pred.shape[0]):
                pred_diag = pred_diagrams[i]
                target_diag = target_diagrams[i]
                
                if pred_diag is not None and target_diag is not None:
                    d0,d1=compute_wasserstein_distance(pred_diag,target_diag)
                    pass
                elif pred_diag is not None:
                    d0,d1=compute_total_persistence(pred_diag)
                    pass
                elif target_diag is not None:
                    d0,d1=compute_total_persistence(target_diag)
                    pass
                else:
                    d0,d1=0,0
                    pass
                d0_total+=d0
                d1_total+=d1
        omega = 1 + self.alpha * (d0_total / pred.shape[0]*2) + self.beta * (d1_total / pred.shape[0]*2)
        if np.isinf(d0_total) or np.isinf(d1_total):
            print("WARNING: Infinity detected!")
        if omega > 100:
            print("WARNING: omega very large!")

        return omega


def get_persistence_info(prob_volume):
    """
    Returns list of dicts, each containing:
    - 'birth_coords': (z, y, x) tuple
    - 'death_coords': (z, y, x) tuple or None if infinite
    - 'birth_value': float
    - 'death_value': float or inf
    - 'dim': 0, 1, or 2 (which homology dimension)
    """
    cc = gudhi.CubicalComplex(top_dimensional_cells=-prob_volume)
    cc.persistence()
    pairs = cc.cofaces_of_persistence_pairs()
    
    result = []
    
    # Finite pairs: pairs[0][dim] contains array of [birth_idx, death_idx]
    for dim, dim_pairs in enumerate(pairs[0]):
        for pair in dim_pairs:
            birth_idx, death_idx = pair[0], pair[1]
            birth_coords = np.unravel_index(birth_idx, prob_volume.shape)
            death_coords = np.unravel_index(death_idx, prob_volume.shape)
            
            result.append({
                'birth_coords': birth_coords,
                'death_coords': death_coords,
                'birth_value': prob_volume[birth_coords],
                'death_value': prob_volume[death_coords],
                'dim': dim
            })
    
    # Infinite pairs: pairs[1][dim] contains array of [birth_idx] only
    for dim, dim_pairs in enumerate(pairs[1]):
        for birth_idx in dim_pairs:
            birth_coords = np.unravel_index(birth_idx, prob_volume.shape)
            result.append({
                'birth_coords': birth_coords,
                'death_coords': None,
                'birth_value': prob_volume[birth_coords],
                'death_value': np.inf,
                'dim': dim
            })
    return result



def match_diagrams(pred_info, target_info, dims: int = 3):
    """
    pred_info: list of dicts from get_persistence_info(pred_volume)
    target_info: list of dicts from get_persistence_info(target_volume)
    """
    matches = []
    
    for dim in range(dims):
        # Include ALL features (finite AND infinite)
        pred_dim = [p for p in pred_info if p['dim'] == dim]
        target_dim = [t for t in target_info if t['dim'] == dim]
        
        n_pred = len(pred_dim)
        n_target = len(target_dim)
        
        if n_pred == 0 and n_target == 0:
            continue
        if n_pred == 0:
            for t in target_dim:
                matches.append((None, t))
            continue
        if n_target == 0:
            for p in pred_dim:
                matches.append((p, None))
            continue
        
        size = n_pred + n_target
        cost = np.full((size, size), np.inf)
        
        # Helper to get effective death value (treat inf as 0 for matching cost) (this part was claude)
        def effective_death(dot):
            if dot['death_value'] == np.inf:
                return 0.0  # Convention: infinite features "die at 0"
            return dot['death_value']
        
        # 1. Pred-to-target distances
        for i, pred_dot in enumerate(pred_dim):
            for j, target_dot in enumerate(target_dim):
                dist = (pred_dot['birth_value'] - target_dot['birth_value'])**2 + \
                       (effective_death(pred_dot) - effective_death(target_dot))**2
                cost[i, j] = dist
        
        # 2. Pred-to-diagonal costs
        for i, pred_dot in enumerate(pred_dim):
            d_eff = effective_death(pred_dot)
            diag_cost = ((pred_dot['birth_value'] - d_eff) ** 2) / 2
            cost[i, n_target + i] = diag_cost
        
        # 3. Target-to-diagonal costs
        for j, target_dot in enumerate(target_dim):
            d_eff = 0.0 if target_dot['death_value'] == np.inf else target_dot['death_value']
            diag_cost = ((target_dot['birth_value'] - d_eff) ** 2) / 2
            
            # Penalty: infinite features should not go to diagonal easily
            if target_dot['death_value'] == np.inf:
                diag_cost += 1.0  # Add penalty - forces matching if possible
            
            cost[n_pred + j, j] = diag_cost

        # for j, target_dot in enumerate(target_dim):
        #     d_eff = effective_death(target_dot)
        #     diag_cost = ((target_dot['birth_value'] - d_eff) ** 2) / 2
        #     cost[n_pred + j, j] = diag_cost
        
        # 4. Diagonal-to-diagonal - free
        cost[n_pred:, n_target:] = 0
        
        row_ind, col_ind = linear_sum_assignment(cost)
        
        for r, c in zip(row_ind, col_ind):
            if r < n_pred and c < n_target:
                matches.append((pred_dim[r], target_dim[c]))
            elif r < n_pred and c >= n_target:
                matches.append((pred_dim[r], None))
            elif r >= n_pred and c < n_target:
                matches.append((None, target_dim[c]))
    
    return matches

def compute_topo_gradient(matches, prob_volume):
    grad = np.zeros_like(prob_volume)
    
    for pred_dot, target_dot in matches:
        if pred_dot is not None and target_dot is not None:
            # Push birth toward target birth
            grad[pred_dot['birth_coords']] += 2 * (pred_dot['birth_value'] - target_dot['birth_value'])
            # Only push death if finite
            if pred_dot['death_coords'] is not None:
                target_death = 0.0 if target_dot['death_value'] == np.inf else target_dot['death_value']
                grad[pred_dot['death_coords']] += 2 * (pred_dot['death_value'] - target_death)
                
        elif pred_dot is not None and target_dot is None:
            d_eff = 0.0 if pred_dot['death_value'] == np.inf else pred_dot['death_value']
            diagonal_point = (pred_dot['birth_value'] + d_eff) / 2
            grad[pred_dot['birth_coords']] += 2 * (pred_dot['birth_value'] - diagonal_point)
            if pred_dot['death_coords'] is not None:
                grad[pred_dot['death_coords']] += 2 * (pred_dot['death_value'] - diagonal_point)
    
    return grad

class Cubical_Complex_Loss_Differentiable(nn.Module):
    def __init__(self, warmup_epochs: int = 25, classes_wo_background: int = 2, N: int = 1):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.classes_wo_background = classes_wo_background
        self.N = N
        self._target_cache = {}  # Add cache
    
    def _get_cache_key(self, target_np):
        """Create hashable key from target array"""
        return target_np.tobytes()
    
    def forward(self, x, y, current_epoch):
        total_grad = torch.zeros_like(x)
        logging_value = 0.0

        if (current_epoch < self.warmup_epochs) or (current_epoch % self.N != 0):
            return torch.tensor(0.0, device=x.device, requires_grad=False), 0.0
        
        x_soft = F.softmax(x, dim=1)
        x_np = x_soft.detach().cpu().numpy()
        
        tasks = []
        for i in range(y.shape[0]):
            for j in range(self.classes_wo_background):
                target = (y[i] == j + 1).float()
                target_np = target.cpu().numpy()
                pred = x_np[i][j + 1]
                
                # Check cache for target persistence
                cache_key = self._get_cache_key(target_np)
                if cache_key not in self._target_cache:
                    self._target_cache[cache_key] = get_persistence_info(target_np)
                
                target_persistence = self._target_cache[cache_key]
                tasks.append((pred, target_persistence, i, j + 1))
        
        with ThreadPoolExecutor(max_workers=None) as executor:
            results = list(executor.map(_compute_single_topo_cached, tasks))
        
        for res in results:
            total_grad[res['batch_idx']][res['class_idx']] = torch.tensor(res['gradient'], device=x.device)
            logging_value += res['log_value']

        fake_loss = (x_soft * total_grad.detach()).sum()
        return fake_loss, logging_value


def _compute_single_topo_cached(args):
    """Modified to accept pre-computed target persistence"""
    pred, target_persistence, batch_idx, class_idx = args
    
    pred_binary = (pred > 0.5).astype(float)
    pred_persistence = get_persistence_info(pred_binary)
    
    for p in pred_persistence:
        p['birth_value'] = pred[p['birth_coords']]
        if p['death_coords'] is not None:
            p['death_value'] = pred[p['death_coords']]
    
    matches = match_diagrams(pred_info=pred_persistence, target_info=target_persistence)
    topo_grad = compute_topo_gradient(matches=matches, prob_volume=pred)
    logging_value = (topo_grad ** 2).sum()
    
    return {
        'batch_idx': batch_idx,
        'class_idx': class_idx,
        'gradient': topo_grad,
        'log_value': logging_value
    }

