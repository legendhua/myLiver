# -*- coding:utf-8 -*-
"""

Focal loss
鍚屾牱鐢ㄦ潵澶勭悊鍒嗗壊杩囩▼涓殑鍓嶆櫙鑳屾櫙鍍忕礌闈炲钩琛＄殑闂
"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gama=2):
        super().__init__()

        self.alpha = alpha
        self.gama = gama
        self.loss_func = nn.BCELoss(reduce=False)

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)

        # 鏈�鍚庡彇骞冲潎loss鏃堕櫎鐨勭偣涓暟锛岃繖閲屽彧閫夊彇姝ｆ牱鏈�
        normalize_num = torch.numel(target[target > 0])

        # 棣栧厛璁＄畻鏍囧噯鐨勪氦鍙夌喌鎹熷け
        loss = self.loss_func(pred, target)

        exponential_term = ((1 - pred) ** self.gama) * target + (pred ** self.gama) * (1 - target)
        weight_term = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss *= (exponential_term * weight_term)

        return loss.sum() / normalize_num
