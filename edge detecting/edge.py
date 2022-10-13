import torch
import numpy as np
import cv2
from einops import rearrange


for i in range(0,15):
    if i < 10:
        img = cv2.imread("C:\\Users\\lab-com\\Downloads\\WARP_BACK\\VAL_00" + str(i) + "0\\GT.png")
    else:
        img = cv2.imread("C:\\Users\\lab-com\\Downloads\\WARP_BACK\\VAL_0" + str(i) + "0\\GT.png")

    ######### CANNY EDGE DETECTING
    img_t = torch.from_numpy(img).float().permute(2, 0, 1)
    sais_t = rearrange(img_t, "C (H V) (W U) -> C (V H) (U W)", V=4, U=4)
    # sais = sais_t.permute(1, 2, 0).numpy().astype(np.float32)
    sais = sais_t.permute(1,2,0).numpy().astype(np.uint8)
    # sais[sais>1] =1


    # cv2.Canny(img, Lower Threshold, Upper Threshold, Aperture_size(3,5,7, default=3), L2Gradient(boolean, default=true)
    # Threshold: 휘도 변화가 나타나는 정도, lower = 엣지와의 인접부분(선으로얼마나연결?) , higher = 휘도 변화 차이 임계값
    # L2Gradient = sqrt(gradient_x^2 + gradient_y^2) L1Gradient = |grad_x| + |grad_y|

    edge50to100 = cv2.Canny(sais, 50, 100)
    edge50to150 = cv2.Canny(sais, 50, 150)

    ########## LAPLACIAN DETECTING
    src_gray = cv2.cvtColor(sais, cv2.COLOR_BGR2GRAY)
    laplacian3 = cv2.Laplacian(src_gray, cv2.CV_8U, ksize=3)  # 커널사이즈 = 3
    laplacian5 = cv2.Laplacian(src_gray, cv2.CV_8U, ksize=5)  # 커널사이즈 = 5


    if i < 10 :
        cv2.imwrite("./images/canny/00" + str(i) + "0GT_edge50_100.png",edge50to100)
        cv2.imwrite("./images/canny/00" + str(i) + "0GT_edge50_150.png", edge50to150)
        cv2.imwrite("./images/laplacian/00" + str(i) + "0GT_laplacian_3.png",laplacian3)
        cv2.imwrite("./images/laplacian/00" + str(i) + "0GT_laplacian_5.png", laplacian5)
    else :
        cv2.imwrite("./images/canny/0" + str(i) + "0GT_edge50_100.png", edge50to100)
        cv2.imwrite("./images/canny/0" + str(i) + "0GT_edge50_150.png", edge50to150)
        cv2.imwrite("./images/laplacian/0" + str(i) + "0GT_laplacian_3.png", laplacian3)
        cv2.imwrite("./images/laplacian/0" + str(i) + "0GT_laplacian_5.png", laplacian5)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

