import numpy as np

def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray,np.ndarray ]:
    """converts h x w x 3 uint8 RGB imgae into full-resolutin y, cb, cr plane """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected an RGB image with shape (height, width, 3)')
    if rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB values from 0 to 255')

    pixels = rgb.astype(np.float32)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]

    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 128 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 128 + 0.5 * red - 0.418688 * green - 0.081312 * blue

    return np.clip(y, 0, 255), np.clip(cb, 0, 255), np.clip(cr, 0, 255)
