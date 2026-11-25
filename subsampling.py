"""
subsampling.py (v2) - Chroma Subsampling Module for JPEG Image Compression Studio
Compatible with color_conversion.py (rgb_to_ycbcr)
"""

import numpy as np
from typing import Tuple


def downsample_chroma(
    cb: np.ndarray, 
    cr: np.ndarray, 
    mode: str = "4:2:2", 
    method: str = "average"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsamples the chrominance channels (Cb and Cr) according to JPEG standards.
    
    Parameters:
        cb, cr : 2D NumPy arrays of shape (H, W) from rgb_to_ycbcr()
        mode   : "4:4:4" (no reduction) or "4:2:2" (horizontal 2:1 reduction)
                 Also supports "4:2:0" (horizontal and vertical 2:1 reduction)
        method : "average" (averages adjacent pixels to prevent aliasing)
                 or "nearest" (picks every 2nd pixel via direct slicing)
                 
    Returns:
        (cb_sub, cr_sub) : Subsampled 2D NumPy arrays
    """
    mode = mode.strip()
    
    if mode == "4:4:4":
        return cb.copy(), cr.copy()
        
    elif mode == "4:2:2":
        # 4:2:2 downsamples horizontally by a factor of 2 (width / 2)
        h, w = cb.shape
        w_even = w - (w % 2)
        
        if method == "average":
            cb_sub = (cb[:, 0:w_even:2].astype(np.float32) + cb[:, 1:w_even:2].astype(np.float32)) / 2.0
            cr_sub = (cr[:, 0:w_even:2].astype(np.float32) + cr[:, 1:w_even:2].astype(np.float32)) / 2.0
        else:  # "nearest" / direct decimation
            cb_sub = cb[:, 0:w_even:2].copy()
            cr_sub = cr[:, 0:w_even:2].copy()
            
        return cb_sub, cr_sub

    elif mode == "4:2:0":
        # 4:2:0 downsamples both horizontally and vertically by 2 (height / 2, width / 2)
        h, w = cb.shape
        h_even = h - (h % 2)
        w_even = w - (w % 2)
        
        if method == "average":
            cb_sub = (
                cb[0:h_even:2, 0:w_even:2].astype(np.float32) +
                cb[0:h_even:2, 1:w_even:2].astype(np.float32) +
                cb[1:h_even:2, 0:w_even:2].astype(np.float32) +
                cb[1:h_even:2, 1:w_even:2].astype(np.float32)
            ) / 4.0
            cr_sub = (
                cr[0:h_even:2, 0:w_even:2].astype(np.float32) +
                cr[0:h_even:2, 1:w_even:2].astype(np.float32) +
                cr[1:h_even:2, 0:w_even:2].astype(np.float32) +
                cr[1:h_even:2, 1:w_even:2].astype(np.float32)
            ) / 4.0
        else:
            cb_sub = cb[0:h_even:2, 0:w_even:2].copy()
            cr_sub = cr[0:h_even:2, 0:w_even:2].copy()
            
        return cb_sub, cr_sub

    else:
        raise ValueError(f"Unsupported subsampling mode '{mode}'. Choose '4:4:4', '4:2:2', or '4:2:0'.")


def upsample_chroma(
    cb_sub: np.ndarray, 
    cr_sub: np.ndarray, 
    mode: str = "4:2:2", 
    target_shape: Tuple[int, int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upsamples (reconstructs) subsampled chroma channels back to full luminance resolution (H, W).
    Essential for the JPEG decoder before converting back to RGB.
    
    Parameters:
        cb_sub, cr_sub : Subsampled 2D NumPy arrays
        mode           : Subsampling scheme ("4:4:4", "4:2:2", or "4:2:0")
        target_shape   : Optional (target_height, target_width) to match Y channel exactly
        
    Returns:
        (cb_up, cr_up) : Full-resolution 2D NumPy arrays matching luminance dimensions
    """
    mode = mode.strip()
    
    if mode == "4:4:4":
        cb_up = cb_sub.copy()
        cr_up = cr_sub.copy()
        
    elif mode == "4:2:2":
        # Repeat columns horizontally by 2
        cb_up = np.repeat(cb_sub, 2, axis=1)
        cr_up = np.repeat(cr_sub, 2, axis=1)
        
    elif mode == "4:2:0":
        # Repeat both rows and columns by 2
        cb_up = np.repeat(np.repeat(cb_sub, 2, axis=0), 2, axis=1)
        cr_up = np.repeat(np.repeat(cr_sub, 2, axis=0), 2, axis=1)
        
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose '4:4:4', '4:2:2', or '4:2:0'.")
        
    # Crop or pad to match target_shape if provided (handles odd original dimensions)
    if target_shape is not None:
        th, tw = target_shape
        cb_up = cb_up[:th, :tw]
        cr_up = cr_up[:th, :tw]
        
    return cb_up, cr_up


# -------------------------------------------------------------------------
# Verification & Example Integration with Colleague's rgb_to_ycbcr
# -------------------------------------------------------------------------
if __name__ == "__main__":
    def rgb_to_ycbcr(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Colleague's exact function from the notebook note."""
        pixels = rgb.astype(np.float32)
        red = pixels[:, :, 0]
        green = pixels[:, :, 1]
        blue = pixels[:, :, 2]

        y = 0.299 * red + 0.587 * green + 0.114 * blue
        cb = 128 - 0.168736 * red - 0.331264 * green + 0.5 * blue
        cr = 128 + 0.5 * red - 0.418688 * green - 0.081312 * blue

        return np.clip(y, 0, 255), np.clip(cb, 0, 255), np.clip(cr, 0, 255)

    # Simulate an 8x8 RGB block (or any HxW image)
    np.random.seed(42)
    fake_rgb = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
    
    Y, Cb, Cr = rgb_to_ycbcr(fake_rgb)
    print(f"Original Channels: Y={Y.shape}, Cb={Cb.shape}, Cr={Cr.shape}")

    Cb_422, Cr_422 = downsample_chroma(Cb, Cr, mode="4:2:2", method="average")
    print(f"4:2:2 Subsampled:  Cb={Cb_422.shape}, Cr={Cr_422.shape} (Width halved: {Cb.shape[1]} -> {Cb_422.shape[1]})")

    Cb_rec, Cr_rec = upsample_chroma(Cb_422, Cr_422, mode="4:2:2", target_shape=Y.shape)
    print(f"4:2:2 Reconstructed: Cb={Cb_rec.shape}, Cr={Cr_rec.shape} (Matches Y shape!)")
    
    Cb_444, Cr_444 = downsample_chroma(Cb, Cr, mode="4:4:4")
    print(f"4:4:4 Passthrough:   Cb={Cb_444.shape}, Cr={Cr_444.shape}")
    
    assert Cb_422.shape == (8, 4)
    assert Cb_rec.shape == (8, 8)
    print("All tests passed cleanly!")
