# Taken from: https://ieeexplore.ieee.org/abstract/document/9578225/
import numpy as np
import cv2
import torch
import torch.nn.functional as F

def opencv_skelitonize(img):
    skel = np.zeros(img.shape, np.uint8)
    img = img.astype(np.uint8)
    size = np.size(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel

def dice_loss(pred, target):
    '''
    inputs shape  (batch, channel, height, width).
    calculate dice loss per batch and channel of sample.
    E.g. if batch shape is [64, 1, 128, 128] -> [64, 1]
    '''
    smooth = 1.
    iflat = pred.view(*pred.shape[:2], -1) #batch, channel, -1
    tflat = target.view(*target.shape[:2], -1)
    intersection = (iflat * tflat).sum(-1)
    return -((2. * intersection + smooth) /
              (iflat.sum(-1) + tflat.sum(-1) + smooth))

def soft_skeletonize_3d(x, thresh_width=10):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool3d(x*-1, (3,3, 3), 1, (1,1,1))*-1
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool3d(min_pool_x, (3,3, 3), 1, (1,1,1)) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x

def norm_intersection(center_line, vessel):
    '''
    inputs shape  (batch, channel, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    '''
    smooth = 1.
    clf = center_line.view(*center_line.shape[:2], -1)
    vf = vessel.view(*vessel.shape[:2], -1)
    intersection = (clf * vf).sum(-1)
    return (intersection + smooth) / (clf.sum(-1) + smooth)

def soft_cldice_loss(pred, target,class_weights=None,thresh_width=10):
    '''
    inputs shape  (batch, channel, height, width).
    calculate clDice loss
    Because pred and target at moment of loss calculation will be a torch tensors
    it is preferable to calculate target_skeleton on the step of batch forming,
    when it will be in numpy array format by means of opencv
    Assuming class 0 is background
    And averages all classes into one loss
    '''
    total_loss=0
    num_classes = pred.shape[1]
    
    # Convert pred logits to probabilities
    pred_soft = F.softmax(pred, dim=1)

    if class_weights is None:
        class_weights = [1.0] * (num_classes - 1)
    
    # Convert target to one-hot: [B, D, H, W] -> [B, C, D, H, W]
    target_onehot = F.one_hot(target, num_classes)  # [B, D, H, W, C]
    target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]

    for i in range(num_classes - 1):
        class_idx = i + 1  # Actual class index (1 for nerve, 2 for vessel)
        
        pred_class = pred_soft[:, class_idx:class_idx+1, :, :, :]      # [B, 1, D, H, W]
        target_class = target_onehot[:, class_idx:class_idx+1, :, :, :] # [B, 1, D, H, W]
        
        # Skeletonize both
        skel_pred = soft_skeletonize_3d(pred_class, thresh_width)
        skel_target = soft_skeletonize_3d(target_class, thresh_width)
        
        # Topology precision: pred skeleton inside target vessel
        tprec = norm_intersection(skel_pred, target_class)
        
        # Topology sensitivity: target skeleton inside pred vessel  
        tsens = norm_intersection(skel_target, pred_class)
        
        # clDice for this class
        cldice = (2. * tprec * tsens) / (tprec + tsens + 1e-7)
        
        # Weight and accumulate (cldice is similarity, so loss = 1 - cldice)
        total_loss += class_weights[i] * (1.0 - cldice)
    
    return total_loss.mean()