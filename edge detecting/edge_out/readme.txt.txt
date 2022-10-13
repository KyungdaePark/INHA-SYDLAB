22.10.13

Canny : Threshold = (50,100) & (50,150)
Laplacian : Kernel = 3 & 5
Hed

Canny(50,100) * Laplacian3 = canny_laplacian_50_100_k3
Canny(50,150) * Laplacian3 = canny_laplacian_50_150_k3
Canny(50,100) * Laplacian5 = canny_laplacian_50_100_k5
Canny(50,150) * Laplacian5 = canny_laplacian_50_150_k5

Canny(50,100) * hed = canny_hed1_50_100
Canny(50,150) * hed = canny_hed1_50_150

Laplacian(k=3) * hed = laplacian_hed1_k3
Laplacian(k=5) * hed = laplacian_hed1_k5

origin : GT Image
origin_16 : 16 GT Image
hed1 : hed without *upscaling
hed2 : hed with *scaling 
hed_16 : origin_16 using pytorch-hed
hed : origin using pytorch-hed
canny : origin_16 * canny (50,100) & (50,150)
laplacian : origin_16 * laplacian (k=3) & (k=5)

*upscaling : 
	hed[hed>=0.5] = 1
	hed[hed<0.5] = 0