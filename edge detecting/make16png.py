import torch
import numpy as np
import cv2
import pandas as pd
from einops import rearrange

for i in range(0,15):
    if i < 10:
        img = cv2.imread("C:\\Users\\lab-com\\Downloads\\WARP_BACK\\VAL_00" + str(i) + "0\\GT.png")
    else:
        img = cv2.imread("C:\\Users\\lab-com\\Downloads\\WARP_BACK\\VAL_0" + str(i) + "0\\GT.png")

    img_t = torch.from_numpy(img).float().permute(2, 0, 1)
    sais_t = rearrange(img_t, "C (H V) (W U) -> C (V H) (U W)", V=4, U=4)
    sais = sais_t.permute(1, 2, 0).numpy().astype(np.float32)

    cv2.imwrite("GT" + str(i) + "_16.png", sais)