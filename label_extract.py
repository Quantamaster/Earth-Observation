import rasterio
import rasterio.windows
from scipy.stats import mode
import pandas as pd
import numpy as np

# Load the result from your previous filtering
imgs_within_grid = pd.read_csv("data/imgs_within_grid.csv")

# Quick inspection/limiting for a development run
imgs_within_grid = imgs_within_grid.head(100)  # Remove .head for full dataset

# Open land cover raster
lc_ds = rasterio.open("data/worldcover_bbox_delhi_ncr_2021.tif")


def extract_patch_labels(row, patch_size=128):
    """
    Extract land cover patch from raster at given coordinates.

    Args:
        row: DataFrame row containing 'lon' and 'lat' columns
        patch_size: Size of square patch to extract (default 128)

    Returns:
        tuple: (patch_array, window) or (None, None) if extraction fails
    """
    try:
        x, y = row['lon'], row['lat']
        row_idx, col_idx = lc_ds.index(x, y)

        # Calculate window offsets
        row_off = row_idx - patch_size // 2
        col_off = col_idx - patch_size // 2

        # Comprehensive bounds checking
        if (row_off < 0 or col_off < 0 or
                row_off + patch_size > lc_ds.height or
                col_off + patch_size > lc_ds.width):
            return None, None

        # Create and read window
        window = rasterio.windows.Window(col_off, row_off, patch_size, patch_size)
        patch = lc_ds.read(1, window=window)
        return patch, window

    except Exception as e:
        # Log the error if needed: print(f"Error extracting patch: {e}")
        return None, None


# Apply patch extraction to all image chips
imgs_within_grid['lc_patch'] = imgs_within_grid.apply(
    lambda row: extract_patch_labels(row)[0], axis=1)


def get_mode_label(patch):
    """
    Calculate the mode (most frequent value) from a patch, excluding NoData pixels.

    Args:
        patch: NumPy array representing the land cover patch

    Returns:
        Mode value or np.nan if calculation fails
    """
    if patch is None:
        return np.nan

    # Create mask for valid (non-NoData) pixels
    mask = patch != lc_ds.nodata

    if np.sum(mask) == 0:
        return np.nan

    try:
        # Handle both old and new scipy.stats.mode formats
        result = mode(patch[mask].flatten(), keepdims=True)

        # Extract mode value - works for both scipy versions
        if hasattr(result, 'mode'):
            return result.mode[0]
        else:
            return result[0][0]

    except Exception as e:
        # Handle any unexpected errors
        return np.nan


# Calculate dominant land cover labels
imgs_within_grid['label'] = imgs_within_grid['lc_patch'].apply(get_mode_label)

# ESA WorldCover class mapping
ESA_CODE_TO_LABEL = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse",
    70: "Snow/ice",
    80: "Water",
    90: "Wetland",
    95: "Mangrove",
    100: "Moss/lichen",
}

imgs_within_grid['class_str'] = imgs_within_grid['label'].map(ESA_CODE_TO_LABEL)


def edge_case_treatment(patch):
    """
    Determine if a patch has sufficient valid data for reliable classification.

    Args:
        patch: NumPy array representing the land cover patch

    Returns:
        bool: True if patch is valid, False otherwise
    """
    if patch is None:
        return False

    n_total = patch.size
    n_valid = np.sum(patch != lc_ds.nodata)

    # Require at least 50% valid pixels
    if n_valid == 0 or (n_valid / n_total) < 0.5:
        return False

    return True


# Apply validity filtering
imgs_within_grid['valid'] = imgs_within_grid['lc_patch'].apply(edge_case_treatment)
imgs_clean = imgs_within_grid[imgs_within_grid['valid']].reset_index(drop=True)

print(f"Valid images after filtering: {len(imgs_clean)}")
print(f"Class distribution:")
print(imgs_clean['class_str'].value_counts())

# Optional: Save the cleaned dataset
# imgs_clean.to_csv("data/labelled_images_clean.csv", index=False)
