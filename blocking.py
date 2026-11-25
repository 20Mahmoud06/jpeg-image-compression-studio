"""
blocking.py
JPEG Image Compression Studio

Member 05:
Blocking + Level Shifting

Pipeline:
    Y, Cb, Cr
        |
        v
    Padding
        |
        v
    Divide into 8x8 Blocks
        |
        v
    Level Shifting (-128)
        |
        v
    FDCT (Member 06)


Input:
    Three 2D NumPy arrays:
        Y
        Cb
        Cr

Output:
    A dictionary containing:
        - Processed blocks for Y
        - Processed blocks for Cb
        - Processed blocks for Cr
        - Original dimensions
        - Padded dimensions

Each block is an 8x8 NumPy array,
ready to be passed to forward_dct().
"""

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# JPEG uses 8x8 blocks.
BLOCK_SIZE = 8
# Level shifting value.
# Pixel values: [0, 255]
# Shifted values: [-128, 127]

LEVEL_SHIFT = 128

# Padding strategy:
#
# "edge":
# Repeat the last row/column to complete the block.
#
# "constant":
# Fill missing pixels with PADDING_CONSTANT.
#
# TODO:
# Confirm the padding strategy with the team.
PADDING_MODE = "edge"

# Used only when PADDING_MODE = "constant".
PADDING_CONSTANT = 0


# ============================================================
# 1. VALIDATE INPUT
# ============================================================

def validate_channel(channel, channel_name="Channel"):
    """
    Validate that the input is a valid 2D image channel.

    Parameters:
        channel: NumPy array representing an image channel.
        channel_name: Name of the channel (Y, Cb, or Cr).

    Returns:
        A float32 NumPy array.
    """

    channel = np.asarray(channel)

    if channel.ndim != 2:
        raise ValueError(
            f"{channel_name} must be a 2D array."
        )

    if channel.size == 0:
        raise ValueError(
            f"{channel_name} cannot be empty."
        )

    if not np.issubdtype(channel.dtype, np.number):
        raise TypeError(
            f"{channel_name} must contain numeric values."
        )

    if not np.all(np.isfinite(channel)):
        raise ValueError(
            f"{channel_name} contains invalid values."
        )

    return channel.astype(np.float32)


# ============================================================
# 2. PADDING
# ============================================================

def pad_channel(
    channel,
    block_size=BLOCK_SIZE,
    mode=PADDING_MODE
):
    """
    Pad an image channel so that its dimensions
    are divisible by block_size.

    Parameters:
        channel:
            2D NumPy array.

        block_size:
            Block dimension (default: 8).

        mode:
            "edge" or "constant".

    Returns:
        padded_channel:
            Padded image channel.

        original_shape:
            Original (height, width).

        padded_shape:
            New (height, width).

    TODO:
        Confirm the padding method with the team.
    """

    if block_size <= 0:
        raise ValueError("Block size must be positive.")

    height, width = channel.shape

    # Calculate the required padding.
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size

    padding = (
        (0, pad_height),
        (0, pad_width)
    )

    if mode == "edge":

        padded_channel = np.pad(
            channel,
            padding,
            mode="edge"
        )

    elif mode == "constant":

        padded_channel = np.pad(
            channel,
            padding,
            mode="constant",
            constant_values=PADDING_CONSTANT
        )

    else:

        raise ValueError(
            f"Unsupported padding mode: {mode}. "
            "Choose 'edge' or 'constant'."
        )

    return (
        padded_channel,
        (height, width),
        padded_channel.shape
    )


# ============================================================
# 3. BLOCKING
# ============================================================

def split_into_blocks(
    channel,
    block_size=BLOCK_SIZE
):
    """
    Divide a 2D image channel into non-overlapping blocks.

    Parameters:
        channel:
            2D NumPy array.
            Its dimensions must be divisible by block_size.

        block_size:
            Block dimension (default: 8).

    Returns:
        blocks:
            List of 2D NumPy arrays.

            Each block has shape (8, 8).

    Ordering:
        Row-major order.

        Example:
            Block 1 -> Top-left
            Block 2 -> Top-right
            Block 3 -> Bottom-left
            Block 4 -> Bottom-right
    """

    height, width = channel.shape

    if height % block_size != 0 or width % block_size != 0:

        raise ValueError(
            "Channel dimensions must be divisible "
            "by block_size. Apply padding first."
        )

    blocks = []

    for row in range(0, height, block_size):

        for col in range(0, width, block_size):

            block = channel[
                row:row + block_size,
                col:col + block_size
            ]

            blocks.append(block.copy())

    return blocks


# ============================================================
# 4. LEVEL SHIFTING
# ============================================================

def level_shift(block, shift_value=LEVEL_SHIFT):
    """
    Apply JPEG level shifting to a block.

    Formula:
        shifted_pixel = original_pixel - 128

    Parameters:
        block:
            2D NumPy array.

        shift_value:
            Value to subtract (default: 128).

    Returns:
        shifted_block:
            Float32 NumPy array.

    Important:
        Conversion to float32 happens before subtraction
        to avoid unsigned integer underflow.
    """

    block = np.asarray(block, dtype=np.float32)

    shifted_block = block - shift_value

    return shifted_block


# ============================================================
# 5. PROCESS ONE CHANNEL
# ============================================================

def process_channel(
    channel,
    channel_name="Channel",
    block_size=BLOCK_SIZE,
    shift_value=LEVEL_SHIFT,
    padding_mode=PADDING_MODE
):
    """
    Complete processing for one image channel.

    Steps:
        1. Validate input.
        2. Apply padding.
        3. Divide into blocks.
        4. Apply level shifting to each block.

    Returns:
        Dictionary containing:

            blocks:
                List of shifted 8x8 blocks.

            original_shape:
                Original channel dimensions.

            padded_shape:
                Dimensions after padding.

            number_of_blocks:
                Total number of blocks.
    """

    channel = validate_channel(
        channel,
        channel_name
    )

    # Step 1: Padding
    padded_channel, original_shape, padded_shape = pad_channel(
        channel,
        block_size,
        padding_mode
    )

    # Step 2: Blocking
    blocks = split_into_blocks(
        padded_channel,
        block_size
    )

    # Step 3: Level Shifting
    shifted_blocks = [
        level_shift(block, shift_value)
        for block in blocks
    ]

    return {
        "blocks": shifted_blocks,
        "original_shape": original_shape,
        "padded_shape": padded_shape,
        "number_of_blocks": len(shifted_blocks)
    }


# ============================================================
# 6. PROCESS COMPLETE IMAGE
# ============================================================

def process_image(
    y,
    cb,
    cr,
    block_size=BLOCK_SIZE,
    shift_value=LEVEL_SHIFT,
    padding_mode=PADDING_MODE
):
    """
    Process the three YCbCr channels.

    Input:
        y:
            Y channel (2D NumPy array).

        cb:
            Cb channel (2D NumPy array).
            Can be subsampled.

        cr:
            Cr channel (2D NumPy array).
            Can be subsampled.

    Output:
        Dictionary containing processed results
        for all three channels.

    IMPORTANT:
        Y, Cb, and Cr are processed independently.

        This is necessary because Cb and Cr may have
        different dimensions after chroma subsampling.

    TODO:
        Confirm the expected output structure
        with the FDCT member.
    """

    # Process Y channel
    y_result = process_channel(
        y,
        channel_name="Y",
        block_size=block_size,
        shift_value=shift_value,
        padding_mode=padding_mode
    )

    # Process Cb channel
    cb_result = process_channel(
        cb,
        channel_name="Cb",
        block_size=block_size,
        shift_value=shift_value,
        padding_mode=padding_mode
    )

    # Process Cr channel
    cr_result = process_channel(
        cr,
        channel_name="Cr",
        block_size=block_size,
        shift_value=shift_value,
        padding_mode=padding_mode
    )

    return {
        "Y": y_result,
        "Cb": cb_result,
        "Cr": cr_result
    }


# ============================================================
# 7. SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    # Create sample YCbCr channels.
    # These dimensions are intentionally different
    # to simulate chroma subsampling.

    y = np.random.randint(
        0, 256, (16, 16)
    ).astype(np.float32)

    cb = np.random.randint(
        0, 256, (8, 8)
    ).astype(np.float32)

    cr = np.random.randint(
        0, 256, (8, 8)
    ).astype(np.float32)

    # Process all channels.
    results = process_image(y, cb, cr)

    # Display results.
    for channel_name, result in results.items():

        print(f"\n{channel_name} Channel")

        print("Original shape:", result["original_shape"])

        print("Padded shape:", result["padded_shape"])

        print("Number of blocks:", result["number_of_blocks"])

        print("First block after level shifting:")

        print(result["blocks"][0])

        print("Block shape:", result["blocks"][0].shape)

    print("\nAll processing completed successfully!")