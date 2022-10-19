from PIL import Image
import numpy as np
import cv2

def get_N(W, H):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_N_inv( W, H):
    """N that maps from normalized to unnormalized coordinates"""
    # TODO: do this analytically maybe?
    N = get_N(W, H)
    return np.linalg.inv(N)

def cvt_MToTheta(M, w, h):
    """convert affine warp matrix `M` compatible with `opencv.warpAffine` to `theta` matrix
    compatible with `torch.F.affine_grid`

    Parameters
    ----------
    M : np.ndarray
        affine warp matrix shaped [2, 3]
    w : int
        width of image
    h : int
        height of image

    Returns
    -------
    np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    """
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def _get_border( border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

local_file_path = "000000330818.jpg"

scale_range = [0.6, 1.4]
shift = 0
rot = 0
target_wh = [512, 512]
output_file = 'affined_image'
with open(local_file_path, 'rb') as f:
    image = Image.open(f).convert('RGB')

np.random.seed(123)
center = np.array([image.size[1] / 2., image.size[0] / 2.], dtype=np.float32)
s = max(image.size[1], image.size[0]) * 1.0
scale = s*np.random.choice(np.arange(scale_range[0], scale_range[1], 0.1))
w_border = _get_border(128, image.size[1])
h_border = _get_border(128, image.size[0])
#center[0] = np.random.randint(low=w_border, high=image.size[1] - w_border)
#center[1] = np.random.randint(low=h_border, high=image.size[0] - h_border)




if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
    scale = np.array([scale, scale], dtype=np.float32)
if isinstance(scale, list):
    scale = np.array(scale)

scale_tmp = scale
src_w = scale_tmp[0]
dst_w = target_wh[0]
dst_h = target_wh[1]

rot_rad = np.pi * rot / 180
src_dir = get_dir([0, src_w * -0.5], rot_rad)
dst_dir = np.array([0, dst_w * -0.5], np.float32)

src = np.zeros((3, 2), dtype=np.float32)
dst = np.zeros((3, 2), dtype=np.float32)
src[0, :] = center + scale_tmp * shift
src[1, :] = center + src_dir + scale_tmp * shift
dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

src[2:, :] = get_3rd_point(src[0, :], src[1, :])
dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

#import pdb; pdb.set_trace()
trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
image_cv = cv2.warpAffine(np.array(image)[:, :, ::-1].copy(), trans,
                     (dst_w, dst_h),
                     flags=cv2.INTER_LINEAR)

cv2.imwrite(output_file + "_cv.jpeg",  image_cv)
trans = cv2.invertAffineTransform(trans)
image_pil = image.transform(
        (dst_w, dst_h),
        Image.AFFINE,
        #theta.flatten(),
        trans.flatten(),
        resample=Image.BILINEAR
    )
image_pil.save(output_file + "_pil.jpeg", "JPEG")
