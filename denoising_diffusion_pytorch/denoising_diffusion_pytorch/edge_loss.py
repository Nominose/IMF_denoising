import torch.nn.functional as F

def edge_loss_fn(pred, target):
    
    # 计算 x 方向和 y 方向的梯度
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    # 使用 L1 loss（相比 L2 更适合边缘）
    loss_x = F.l1_loss(pred_dx, target_dx)
    loss_y = F.l1_loss(pred_dy, target_dy)
    return loss_x + loss_y