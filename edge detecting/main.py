import torch
import numpy as np
import cv2
from einops import rearrange

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from pyhed import temp

# print(temp.getTenoutput("./GT0_16.png"))

for i in range(0,15):
    if i < 10:
        img = cv2.imread("C:\\Users\\lab-com\\Downloads\\WARP_BACK\\VAL_00" + str(i) + "0\\GT.png")

    else:
        img = cv2.imread("C:\\Users\\lab-com\\Downloads\\WARP_BACK\\VAL_0" + str(i) + "0\\GT.png")
    src_hed = "./GT" + str(i) + "_16.png"

    img_t = torch.from_numpy(img).float().permute(2, 0, 1)
    sais_t = rearrange(img_t, "C (H V) (W U) -> C (V H) (U W)", V=4, U=4)
    sais = sais_t.permute(1, 2, 0).numpy().astype(np.uint8) # float32를 canny에 돌릴 수 없어서 uint8


    ## CANNY
    canny50_100 = cv2.Canny(sais,50,100) # defualt canny result
    canny50_150 = cv2.Canny(sais,50,150)

    ## LAPLACIAN
    src_gray = cv2.cvtColor(sais, cv2.COLOR_BGR2GRAY)
    laplacian3 = cv2.Laplacian(src_gray, cv2.CV_8U, ksize=3)
    laplacian5 = cv2.Laplacian(src_gray, cv2.CV_8U, ksize=5)

    #HED
    hed = temp.getTenoutput(src_hed)
    hed_numpy1 = hed.numpy().reshape(1024,1024)



    # 안하는게 더 정확한 결과?
    # hed upscaling for float32 -> uint8 (np logical and)
    # hed 값은 0.00xxx...
    # hed의 모든 요소에 *255 : hed_numpy1
    # hed값이 특정 임계값(=0.5) 이상이라면 1 혹은 0 표현 : hed_numpy2
    # hed_numpy1[hed_numpy1>0] *= 255


    # canny * laplacian
    andnp50_100_3 = np.logical_and(canny50_100, laplacian3).astype(np.float32)
    andnp50_150_3 = np.logical_and(canny50_150, laplacian3).astype(np.float32)
    andnp50_100_5 = np.logical_and(canny50_100, laplacian5).astype(np.float32)
    andnp50_150_5 = np.logical_and(canny50_150, laplacian5).astype(np.float32)

    # canny * hed
    andnp50_100_h1 = np.logical_and(canny50_100, hed_numpy1).astype(np.float32)
    andnp50_150_h1 = np.logical_and(canny50_150, hed_numpy1).astype(np.float32)

    # laplacian * hed
    andnpl3_h1 = np.logical_and(laplacian3, hed_numpy1).astype(np.float32)
    andnpl5_h1 = np.logical_and(laplacian5, hed_numpy1).astype(np.float32)


    #imwrite를 위한 upscaling
    # [UP] canny * lap
    andnp50_100_3[andnp50_100_3 > 0] *= 255
    andnp50_100_5[andnp50_100_5 > 0] *= 255
    andnp50_150_3[andnp50_150_3 > 0] *= 255
    andnp50_150_5[andnp50_150_5 > 0] *= 255

    # [UP] canny * hed
    andnp50_100_h1[andnp50_100_h1 > 0] *= 255
    andnp50_150_h1[andnp50_150_h1 > 0] *= 255

    # [UP] lap * hed
    andnpl3_h1[andnpl3_h1 > 0] *= 255
    andnpl5_h1[andnpl5_h1 > 0] *= 255

    cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\canny_laplacian_50_100_k3\\andnp50_100_3_" + str(i) + ".png", andnp50_100_3)
    cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\canny_laplacian_50_100_k5\\andnp50_100_5_" + str(i) + ".png", andnp50_100_5)
    cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\canny_laplacian_50_150_k3\\andnp50_150_3_" + str(i) + ".png", andnp50_150_3)
    cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\canny_laplacian_50_150_k5\\andnp50_100_5_" + str(i) + ".png", andnp50_150_5)

    cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\canny_hed1_50_100\\andnp50_100_h_" + str(i) + ".png", andnp50_100_h1)
    cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\canny_hed1_50_150\\andnp50_150_h_" + str(i) + ".png", andnp50_150_h1)

    cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\laplacian_hed1_k3\\andnpl3_h_" + str(i) + ".png", andnpl3_h1)
    cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\laplacian_hed1_k5\\andnpl5_h_" + str(i) + ".png", andnpl5_h1)


    # # hed2 upscaling 했을때와 안했을 때 비교, numpy1이 upscaling 안했을때
    # hed_numpy2 = hed_numpy1.astype(np.float32)
    # hed_numpy2[hed_numpy2 >= 0.5] = 1
    # hed_numpy2[hed_numpy2 < 0.5] = 0
    #
    # hed_numpy1[hed_numpy1 > 0] *= 255
    # hed_numpy2[hed_numpy2 > 0] *= 255
    # cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\hed1\\" + str(i) + ".png", hed_numpy1)
    # cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\hed2\\" + str(i) + ".png", hed_numpy2)



    #imshow
    # cv2.imshow("canny50100", canny50_100)
    # cv2.imshow("laplaciank3", laplacian3)
    # cv2.imshow("hed", hed_numpy)
    # cv2.imshow("canny*lap", andnp50_100_3)
    # cv2.imshow("canny*hed", andnp50_100_h)
    # cv2.imshow("lap*hed",andnpl3_h)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # andnp50_100_h2 = np.logical_and(canny50_100, hed_numpy2).astype(np.float32)
    # andnp50_150_h2 = np.logical_and(canny50_150, hed_numpy2).astype(np.float32)
    # andnpl3_h2 = np.logical_and(laplacian3, hed_numpy2).astype(np.float32)
    # andnpl5_h2 = np.logical_and(laplacian5, hed_numpy2).astype(np.float32)

    # andnp50_100_h2[andnp50_100_h2 > 0] *= 255
    # andnp50_150_h2[andnp50_150_h2 > 0] *= 255
    # andnpl3_h2[andnpl3_h2 > 0] *= 255
    # andnpl5_h2[andnpl5_h2 > 0] *= 255

    # cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\canny_hed2_50_100\\andnp50_100_h2" + str(i) + ".png",andnp50_100_h2)
    # cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\canny_hed2_50_150\\andnp50_150_h2" + str(i) + ".png",andnp50_150_h2)
    # cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\laplacian_hed2_k3\\andnpl3_h2" + str(i) + ".png", andnpl3_h2)
    # cv2.imwrite("C:\\Users\\lab-com\\Documents\\out\\laplacian_hed2_k5\\andnpl5_h2" + str(i) + ".png", andnpl5_h2)