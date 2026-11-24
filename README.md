# JPEG Image Compression Studio

An academic Digital Image Processing project in Python with a PyQt6 interface. The team is building the JPEG compression pipeline one stage at a time.

## RGB to YCbCr conversion

Yousif Elkattawy implemented the color conversion stage. It takes an 8-bit RGB image and separates luma from color information.

| Channel | What it represents |
| --- | --- |
| `Y` | Luma, a weighted combination of red, green, and blue that approximates perceived brightness |
| `Cb` | Blue difference, based on the difference between blue and luma |
| `Cr` | Red difference, based on the difference between red and luma |

The conversion uses NumPy to apply these formulas to every pixel:

```text
Y  =       0.299000 R + 0.587000 G + 0.114000 B
Cb = 128 - 0.168736 R - 0.331264 G + 0.500000 B
Cr = 128 + 0.500000 R - 0.418688 G - 0.081312 B
```

The `128` offset places neutral color at the midpoint of the 8-bit range. For example, a gray pixel with RGB values `(90, 90, 90)` becomes approximately `(90, 128, 128)` in YCbCr.

## Handoff to chroma subsampling

```python
y, cb, cr = rgb_to_ycbcr(rgb)
```

The input is a NumPy array with shape `(height, width, 3)`, dtype `uint8`, and channels in RGB order. The function returns three `float32` arrays, each with shape `(height, width)` and values clipped to `[0, 255]`.
