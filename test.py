#!/usr/bin/env python3
import cv2
import torch

X=torch.arange(0,60).reshape(1,3,4,5)
print(X)
X=X.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()
print(X)
X=X.squeeze(-1).unflatten(0,(1,4,5)).permute(0, 3, 1, 2)
print(X)