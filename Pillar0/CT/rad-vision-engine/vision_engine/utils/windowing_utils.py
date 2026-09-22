"""
Window/level adjustment utilities for medical images.
Provides functions for applying multiple simultaneous windowing strategies.
"""

import torch
import numpy as np
from typing import Union, List, Optional, Dict
import math

# Define anatomical window settings for CT (center/width based)
ANATOMICAL_WINDOWS = {
    "CT": {
        "lung": {"center": -600, "width": 1500},
        "mediastinum": {"center": 50, "width": 400},
        "abdomen": {"center": 40, "width": 400},
        "liver": {"center": 80, "width": 150},
        "bone": {"center": 400, "width": 1800},
        "brain": {"center": 40, "width": 80},
        "subdural": {"center": 75, "width": 215},
        "stroke": {"center": 40, "width": 40},
        "temporal_bone": {"center": 600, "width": 2800},
        "soft_tissue": {"center": 50, "width": 350},
    }
}

# Define percentile-based window settings for X-ray, Mammography, and MR
PERCENTILE_WINDOWS = {
    "XR": {
        "lung": {
            "percentile_min": 2,
            "percentile_max": 50,
            "description": "Emphasizes lung parenchyma and airways",
        },
        "mediastinum": {
            "percentile_min": 30,
            "percentile_max": 80,
            "description": "Emphasizes heart and vessels",
        },
        "bone": {
            "percentile_min": 70,
            "percentile_max": 98,
            "description": "Emphasizes ribs and spine",
        },
        "soft_tissue": {
            "percentile_min": 10,
            "percentile_max": 90,
            "description": "Balanced view of all structures",
        },
    },
    "MG": {
        "standard": {
            "percentile_min": 5,
            "percentile_max": 95,
            "description": "General breast tissue visualization",
        },
        "dense_tissue": {
            "percentile_min": 20,
            "percentile_max": 98,
            "description": "Enhanced visualization of dense tissue",
        },
        "calcifications": {
            "percentile_min": 50,
            "percentile_max": 99.5,
            "description": "Enhanced visualization of calcifications",
        },
        "skin_line": {
            "percentile_min": 0.5,
            "percentile_max": 70,
            "description": "Enhanced visualization of skin line and superficial structures",
        },
        "contrast": {
            "percentile_min": 10,
            "percentile_max": 90,
            "description": "Balanced contrast for overall assessment",
        },
    },
    "MR": {
        "high_contrast": {
            "percentile_min": 20,
            "percentile_max": 99,
            "description": "Maximum tissue contrast",
        }
    },
}


def _get_windows_to_apply(
    windows: Union[str, List[str]] = "default", modality: str = "CT"
) -> List[str]:
    """
    Get list of windows to apply for a given modality.
    """
    if windows == "default":
        if modality == "CT":
            return ["lung", "mediastinum", "bone"]
        elif modality == "MR":
            return ["znorm"]
        elif modality in ["XR", "MG"]:
            return ["standard", "soft_tissue", "high_contrast"]
        else:
            return ["minmax"]
    elif windows == "all":
        return get_available_windows(modality)
    elif isinstance(windows, str):
        return [windows]
    else:
        return windows


def batch_apply_windowing_vectorized(
    volume: Union[torch.Tensor, np.ndarray],
    windows: Union[str, List[str]] = "default",
    modality: str = "CT",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    percentile_windows: Optional[Dict[str, Dict[str, float]]] = None,
    znorm_clip: Optional[float] = None,
    torch_operating_dtype: torch.dtype = torch.float32,
    compute_stats_per_sample: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Batch apply windowing to medical images using vectorized operations.

    This function efficiently applies multiple windowing strategies in parallel
    by treating all windows as shift-and-scale operations.

    Args:
        volume: Input volume as tensor or numpy array.
                Shape: (B, C, D, H, W) or (B, C, H, W) for batched inputs
                       (D, H, W) or (H, W) for single inputs
        windows: Window name(s) to apply (same as apply_windowing)
        modality: Imaging modality ('CT', 'MR', 'XR', 'MG')
        min_value: Override minimum value for minmax windowing
        max_value: Override maximum value for minmax windowing
        percentile_windows: Custom percentile window definitions
        znorm_clip: Clipping value for z-normalized data (e.g., 3.0 clips at ±3 std)
        torch_operating_dtype: Data type for torch operations
        compute_stats_per_sample: If True, compute statistics per sample in batch.
                                 If False, compute across entire batch.

    Returns:
        Windowed volumes. Shape: (B, C*N, D, H, W) where N is number of windows
        Windows and channels are combined into a single dimension for compatibility.
        If input is unbatched, batch dimension is removed appropriately.
    """
    # Convert to tensor if needed
    is_numpy = isinstance(volume, np.ndarray)
    if is_numpy:
        volume = torch.from_numpy(volume)
        # Only convert dtype if necessary
        if volume.dtype != torch_operating_dtype:
            volume = volume.to(torch_operating_dtype)
    else:
        # Only convert dtype if necessary
        if volume.dtype != torch_operating_dtype:
            volume = volume.to(torch_operating_dtype)

    # Handle input shapes - add batch and channel dims if needed
    original_shape = volume.shape
    needs_batch = volume.ndim < 4  # Less than (B, C, H, W)
    needs_channel = volume.ndim < 5 if volume.ndim >= 4 else volume.ndim < 3

    if needs_batch:
        volume = volume.unsqueeze(0)  # Add batch dimension
    if needs_channel:
        volume = volume.unsqueeze(1 if not needs_batch else 0)  # Add channel dimension

    # Ensure we have at least 4 dims (B, C, H, W)
    while volume.ndim < 4:
        volume = volume.unsqueeze(-1)

    B, C = volume.shape[:2]
    spatial_dims = volume.shape[2:]

    # Get list of windows to apply
    windows = _get_windows_to_apply(windows, modality)
    num_windows = len(windows)

    # Compute all window parameters as shift-scale pairs
    window_params = _compute_window_params_vectorized(
        volume,
        windows,
        modality,
        min_value,
        max_value,
        percentile_windows,
        znorm_clip,
        compute_stats_per_sample,
    )  # Shape: (B, num_windows, C, 2)

    # Apply all windows in parallel
    result = _apply_windows_vectorized(
        volume, window_params
    )  # Shape: (B, C*num_windows, *spatial) - already in final layout

    # Handle output shape based on input
    if needs_batch:
        result = result.squeeze(0)  # Remove batch dimension
    if needs_channel and C == 1 and num_windows == 1:
        # For single window, single channel, remove channel dim
        result = result.squeeze(0 if needs_batch else 1)

    # If single window and appropriate shape, remove window dimension
    if num_windows == 1 and needs_batch and needs_channel:
        result = result.squeeze(0)

    # Convert back to numpy if input was numpy
    if is_numpy:
        result = result.cpu().numpy()

    return result


def _compute_window_params_vectorized(
    volume: torch.Tensor,
    windows: List[str],
    modality: str,
    min_value: Optional[float],
    max_value: Optional[float],
    percentile_windows: Optional[Dict[str, Dict[str, float]]],
    znorm_clip: Optional[float],
    compute_stats_per_sample: bool,
) -> torch.Tensor:
    """
    Compute shift and scale parameters for all windows in a vectorized manner.
    Handles multi-channel inputs by computing statistics per channel.

    Returns:
        torch.Tensor: Shape (B, num_windows, C, 2) where [:,:,:,0] is shift and [:,:,:,1] is scale
    """
    B, C = volume.shape[:2]
    spatial_shape = volume.shape[2:]
    num_windows = len(windows)
    device = volume.device
    dtype = volume.dtype

    # Initialize parameters tensor - now includes channel dimension
    params = torch.zeros((B, num_windows, C, 2), device=device, dtype=dtype)

    # Group windows by computation type for efficiency
    percentile_windows_info = []  # [(window_idx, percentile_min, percentile_max)]
    anatomical_indices = []
    minmax_indices = []
    znorm_indices = []

    for i, window in enumerate(windows):
        if window == "minmax":
            minmax_indices.append(i)
        elif window == "znorm":
            znorm_indices.append(i)
        elif window in ANATOMICAL_WINDOWS.get(modality, {}):
            anatomical_indices.append(i)
        elif modality in PERCENTILE_WINDOWS and window in PERCENTILE_WINDOWS[modality]:
            p = PERCENTILE_WINDOWS[modality][window]
            percentile_windows_info.append(
                (i, p["percentile_min"] / 100.0, p["percentile_max"] / 100.0)
            )
        elif percentile_windows and window in percentile_windows:
            p = percentile_windows[window]
            percentile_windows_info.append(
                (i, p["percentile_min"] / 100.0, p["percentile_max"] / 100.0)
            )

    # Compute parameters for anatomical windows (no computation needed)
    # These are fixed values, same for all channels
    for idx in anatomical_indices:
        window_name = windows[idx]
        p = ANATOMICAL_WINDOWS[modality][window_name]
        center = p["center"]
        width = p["width"]
        params[:, idx, :, 0] = center - width / 2  # shift (same for all channels)
        params[:, idx, :, 1] = width  # scale (same for all channels)

    # Compute min/max for minmax windows
    if minmax_indices:
        # Reshape to (B*C, *spatial) to compute per-channel statistics
        vol_reshaped = volume.view(B * C, -1)

        if compute_stats_per_sample:
            # Compute per sample and channel
            if min_value is None:
                min_vals = vol_reshaped.min(dim=1)[0]  # Shape: (B*C,)
            else:
                min_vals = torch.full((B * C,), min_value, device=device, dtype=dtype)

            if max_value is None:
                max_vals = vol_reshaped.max(dim=1)[0]  # Shape: (B*C,)
            else:
                max_vals = torch.full((B * C,), max_value, device=device, dtype=dtype)
        else:
            # Compute across entire batch but per channel
            volume_per_channel = volume.view(C, -1)
            if min_value is None:
                min_vals_per_channel = volume_per_channel.min(dim=1)[0]  # Shape: (C,)
                min_vals = min_vals_per_channel.repeat(B)  # Shape: (B*C,)
            else:
                min_vals = torch.full((B * C,), min_value, device=device, dtype=dtype)

            if max_value is None:
                max_vals_per_channel = volume_per_channel.max(dim=1)[0]  # Shape: (C,)
                max_vals = max_vals_per_channel.repeat(B)  # Shape: (B*C,)
            else:
                max_vals = torch.full((B * C,), max_value, device=device, dtype=dtype)

        # Reshape back to (B, C)
        min_vals = min_vals.view(B, C)
        max_vals = max_vals.view(B, C)

        for idx in minmax_indices:
            params[:, idx, :, 0] = min_vals  # shift
            params[:, idx, :, 1] = (max_vals - min_vals) + 1e-8  # scale

    # Compute z-norm parameters
    if znorm_indices:
        # Reshape to (B*C, *spatial) to compute per-channel statistics
        vol_reshaped = volume.view(B * C, -1)

        if compute_stats_per_sample:
            # Compute per sample and channel
            mean = vol_reshaped.mean(dim=1)  # Shape: (B*C,)
            std = vol_reshaped.std(dim=1)  # Shape: (B*C,)
        else:
            # Compute across entire batch but per channel
            volume_per_channel = volume.view(C, -1)
            mean_per_channel = volume_per_channel.mean(dim=1)  # Shape: (C,)
            std_per_channel = volume_per_channel.std(dim=1)  # Shape: (C,)
            mean = mean_per_channel.repeat(B)  # Shape: (B*C,)
            std = std_per_channel.repeat(B)  # Shape: (B*C,)

        # Reshape back to (B, C)
        mean = mean.view(B, C)
        std = std.view(B, C)

        # Handle zero std
        std = torch.where(std < 1e-8, torch.ones_like(std), std)

        # Set clipping value
        clip_val = znorm_clip if znorm_clip is not None else 3.0

        for idx in znorm_indices:
            # Convert z-norm to shift-scale for mapping [-clip, +clip] to [0, 1]
            params[:, idx, :, 0] = mean - std * clip_val  # shift
            params[:, idx, :, 1] = std * 2 * clip_val  # scale

    # Compute all percentiles in one call
    if percentile_windows_info:
        # Collect all unique percentile values
        all_percentiles = []
        for _, p_min, p_max in percentile_windows_info:
            all_percentiles.extend([p_min, p_max])

        # Always use float for percentile values
        unique_percentiles = torch.tensor(
            sorted(set(all_percentiles)), device=device, dtype=torch.float32
        )

        # Reshape to (B*C, *spatial) to compute per-channel percentiles
        vol_reshaped = volume.view(B * C, -1)

        # Ensure float dtype for quantile computation
        if vol_reshaped.dtype not in [torch.float32, torch.float64]:
            vol_reshaped = vol_reshaped.float()

        # Compute percentiles efficiently with per-channel sampling if needed
        if compute_stats_per_sample:
            # Sample if tensor is too large (to avoid quantile() errors)
            max_elements_per_channel = 2**24  # ~16M elements per channel

            if vol_reshaped.shape[1] > max_elements_per_channel:
                # Sample independently for each channel
                computed_percentiles_list = []
                for bc in range(B * C):
                    # Get this channel's data
                    channel_data = vol_reshaped[bc]
                    # Sample from this channel
                    indices = torch.randint(
                        0,
                        channel_data.shape[0],
                        (max_elements_per_channel,),
                        device=channel_data.device,
                    )
                    channel_sampled = channel_data[indices]
                    # Compute percentiles for this channel
                    channel_percentiles = torch.quantile(
                        channel_sampled, unique_percentiles
                    )
                    computed_percentiles_list.append(channel_percentiles)
                # Stack results
                computed_percentiles = torch.stack(
                    computed_percentiles_list, dim=1
                )  # Shape: (num_unique, B*C)
            else:
                # Small enough to compute directly
                computed_percentiles = torch.quantile(
                    vol_reshaped, unique_percentiles, dim=1
                )  # Shape: (num_unique, B*C)
        else:
            # Compute across entire batch but per channel
            volume_per_channel = volume.view(C, -1)
            max_elements_per_channel = 2**24  # ~16M elements per channel

            # Sample if needed, independently per channel
            if volume_per_channel.shape[1] > max_elements_per_channel:
                computed_percentiles_list = []
                for c in range(C):
                    # Get this channel's data across all batches
                    channel_data = volume_per_channel[c]
                    # Sample from this channel
                    indices = torch.randint(
                        0,
                        channel_data.shape[0],
                        (max_elements_per_channel,),
                        device=channel_data.device,
                    )
                    channel_sampled = channel_data[indices]
                    # Compute percentiles for this channel
                    channel_percentiles = torch.quantile(
                        channel_sampled, unique_percentiles
                    )
                    computed_percentiles_list.append(channel_percentiles)
                # Stack results
                computed_percentiles_per_channel = torch.stack(
                    computed_percentiles_list, dim=1
                )  # Shape: (num_unique, C)
            else:
                # Small enough to compute directly
                computed_percentiles_per_channel = torch.quantile(
                    volume_per_channel, unique_percentiles, dim=1
                )  # Shape: (num_unique, C)

            # Repeat for each batch element
            computed_percentiles = computed_percentiles_per_channel.repeat(
                1, B
            )  # Shape: (num_unique, B*C)

        # Convert back to original dtype
        computed_percentiles = computed_percentiles.to(dtype)

        # Reshape to (num_unique, B, C)
        computed_percentiles = computed_percentiles.view(len(unique_percentiles), B, C)

        # Store percentiles list for direct indexing
        unique_percentiles_list = unique_percentiles.tolist()

        # Assign to params
        for idx, p_min, p_max in percentile_windows_info:
            # Find indices in the unique percentiles list
            low_idx = unique_percentiles_list.index(
                min(unique_percentiles_list, key=lambda x: abs(x - p_min))
            )
            high_idx = unique_percentiles_list.index(
                min(unique_percentiles_list, key=lambda x: abs(x - p_max))
            )

            low = computed_percentiles[low_idx]  # Shape: (B, C)
            high = computed_percentiles[high_idx]  # Shape: (B, C)

            # Avoid division by zero
            scale = high - low

            params[:, idx, :, 0] = low  # shift
            params[:, idx, :, 1] = scale + 1e-8  # scale with epsilon

    return params


def _apply_windows_vectorized(
    volume: torch.Tensor, window_params: torch.Tensor
) -> torch.Tensor:
    """
    Apply all windows in parallel using shift and scale parameters.
    Handles per-channel windowing parameters.

    Args:
        volume: Shape (B, C, *spatial_dims)
        window_params: Shape (B, num_windows, C, 2) - [shift, scale] pairs per channel

    Returns:
        torch.Tensor: Shape (B, C*num_windows, *spatial_dims) - directly in final layout
    """
    B, C = volume.shape[:2]
    num_windows = window_params.shape[1]
    spatial_dims = volume.shape[2:]
    device = volume.device
    dtype = volume.dtype

    # Pre-allocate output in final shape to avoid permute
    result = torch.empty(
        (B, C * num_windows, *spatial_dims), device=device, dtype=dtype
    )

    # Process each window and write directly to correct position
    for w in range(num_windows):
        # Get parameters for this window
        shift = window_params[:, w, :, 0].view(B, C, *([1] * len(spatial_dims)))
        scale = window_params[:, w, :, 1].view(B, C, *([1] * len(spatial_dims)))

        # Apply windowing for this window
        windowed = (volume - shift) / scale
        windowed.clamp_(0, 1)  # In-place clamp

        # Write directly to the correct channels in output
        # Window w, channel c goes to output channel (c * num_windows + w)
        for c in range(C):
            result[:, c * num_windows + w] = windowed[:, c]

    return result


def batch_apply_windowing(
    volume: Union[torch.Tensor, np.ndarray],
    windows: Union[str, List[str]] = "default",
    modality: str = "CT",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    percentile_windows: Optional[Dict[str, Dict[str, float]]] = None,
    znorm_clip: Optional[float] = None,
    torch_operating_dtype: torch.dtype = torch.float32,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Backward compatible wrapper for batch_apply_windowing_vectorized.

    This function calls the vectorized implementation with default settings.
    """
    return batch_apply_windowing_vectorized(
        volume=volume,
        windows=windows,
        modality=modality,
        min_value=min_value,
        max_value=max_value,
        percentile_windows=percentile_windows,
        znorm_clip=znorm_clip,
        torch_operating_dtype=torch_operating_dtype,
        compute_stats_per_sample=True,
    )


def apply_windowing(
    volume: Union[torch.Tensor, np.ndarray],
    windows: Union[str, List[str]] = "default",
    modality: str = "CT",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    percentile_windows: Optional[Dict[str, Dict[str, float]]] = None,
    znorm_clip: Optional[float] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply one or multiple windowing strategies to medical images.

    This function supports:
    1. Standard anatomical windows (center/width based)
    2. Min-max normalization
    3. Percentile-based windows
    4. Z-score normalization (standardization)
    5. Custom window specifications
    6. The 'all' keyword to apply all relevant windows

    Args:
        volume: Input volume as tensor or numpy array. Shape: (D, H, W) or (H, W)
        windows: Window name(s) to apply. Options:
            - Single string: 'lung', 'bone', 'mediastinum', etc.
            - List of strings: ['lung', 'bone'] for multiple windows
            - 'default': Uses modality-specific default
            - 'minmax': Min-max normalization
            - 'znorm': Z-score normalization (recommended for MR)
            - 'all': Applies all available windows for the modality
        modality: Imaging modality ('CT', 'MR', 'XR', 'MG')
        min_value: Override minimum value for minmax windowing
        max_value: Override maximum value for minmax windowing
        percentile_windows: Custom percentile window definitions
        znorm_clip: Clipping value for z-normalized data (e.g., 3.0 clips at ±3 std)

    Returns:
        Windowed volume(s). If single window: same shape as input.
        If multiple windows: shape (N, *input_shape) where N is number of windows

    Examples:
        >>> # Single window
        >>> lung_view = apply_windowing(ct_volume, 'lung')

        >>> # Multiple windows
        >>> views = apply_windowing(ct_volume, ['lung', 'bone', 'mediastinum'])

        >>> # All available windows
        >>> all_views = apply_windowing(ct_volume, 'all')

        >>> # Custom percentile windows
        >>> custom = apply_windowing(volume, 'custom', percentile_windows={
        ...     'custom': {'percentile_min': 5, 'percentile_max': 95}
        ... })
    """
    # Convert to tensor if needed
    is_numpy = isinstance(volume, np.ndarray)
    if is_numpy:
        volume = torch.from_numpy(volume).float()
    else:
        volume = volume.float()

    # Handle default windows
    if windows == "default":
        if modality == "CT":
            windows = ["lung", "mediastinum", "bone"]
        elif modality == "MR":
            # For MR, z-normalization is often the best default
            windows = ["znorm"]
        elif modality in ["XR", "MG"]:
            # For modalities with percentile windows, use standard preset
            windows = ["standard", "soft_tissue", "high_contrast"]
        else:
            windows = "minmax"

    # Handle 'all' keyword
    if windows == "all":
        windows = get_available_windows(modality)

    # Convert single window to list for uniform processing
    if isinstance(windows, str):
        windows = [windows]

    # Apply each window
    results = []
    for window in windows:
        if window == "minmax":
            # Min-max normalization
            if min_value is None:
                min_value = volume.min()
            if max_value is None:
                max_value = volume.max()

            windowed = (volume - min_value) / (max_value - min_value + 1e-8)
            windowed = torch.clamp(windowed, 0, 1)

        elif window == "znorm":
            # Z-score normalization (standardization)
            mean = volume.mean()
            std = volume.std()

            # Avoid division by zero
            if std < 1e-8:
                windowed = torch.zeros_like(volume)
            else:
                windowed = (volume - mean) / std

                # Apply clipping if specified
                if znorm_clip is not None:
                    windowed = torch.clamp(windowed, -znorm_clip, znorm_clip)

                # Scale to [0, 1] for visualization
                # Map [-clip, +clip] to [0, 1], or use default [-3, 3] range
                clip_val = znorm_clip if znorm_clip is not None else 3.0
                windowed = (windowed + clip_val) / (2 * clip_val)
                windowed = torch.clamp(windowed, 0, 1)

        elif window in ANATOMICAL_WINDOWS.get(modality, {}):
            # Standard anatomical window
            params = ANATOMICAL_WINDOWS[modality][window]
            windowed = apply_anatomical_window(
                volume, params["center"], params["width"]
            )

        elif modality in PERCENTILE_WINDOWS and window in PERCENTILE_WINDOWS[modality]:
            # Percentile-based window from predefined
            params = PERCENTILE_WINDOWS[modality][window]
            windowed = _apply_percentile_window(
                volume, params["percentile_min"], params["percentile_max"]
            )

        elif percentile_windows and window in percentile_windows:
            # Custom percentile-based window
            params = percentile_windows[window]
            windowed = _apply_percentile_window(
                volume, params["percentile_min"], params["percentile_max"]
            )

        else:
            raise ValueError(f"Unknown window: {window}")

        results.append(windowed)

    # Stack results if multiple windows
    if len(results) == 1:
        result = results[0]
    else:
        result = torch.stack(results, dim=0)

    # Convert back to numpy if input was numpy
    if is_numpy:
        result = result.numpy()

    return result


def apply_anatomical_window(
    volume: Union[torch.Tensor, np.ndarray], center: float, width: float
) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply traditional center/width windowing to medical images.

    Args:
        volume: Input volume
        center: Window center (level)
        width: Window width

    Returns:
        Windowed volume with values in [0, 1]
    """
    is_numpy = isinstance(volume, np.ndarray)
    if is_numpy:
        volume = torch.from_numpy(volume).float()

    # Calculate window bounds
    min_val = center - width / 2
    max_val = center + width / 2

    # Apply windowing
    windowed = (volume - min_val) / (max_val - min_val + 1e-8)
    windowed = torch.clamp(windowed, 0, 1)

    if is_numpy:
        windowed = windowed.numpy()

    return windowed


def _apply_percentile_window(
    volume: torch.Tensor, percentile_min: float, percentile_max: float
) -> torch.Tensor:
    """Apply percentile-based windowing."""
    # Calculate percentiles
    flat_volume = volume.flatten()
    num_el = flat_volume.numel()
    if num_el >= 2**24:
        flat_volume = flat_volume[:: math.ceil(num_el / 2**24)]
    p_min = torch.quantile(flat_volume, percentile_min / 100.0)
    p_max = torch.quantile(flat_volume, percentile_max / 100.0)

    # Avoid division by zero
    if p_max <= p_min:
        p_max = p_min + 1

    # Apply windowing
    windowed = (volume - p_min) / (p_max - p_min)
    windowed = torch.clamp(windowed, 0, 1)

    return windowed


def apply_multiple_windows(
    volume: Union[torch.Tensor, np.ndarray], modality: str = "CT"
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """
    Apply all relevant windows for a given modality.

    Args:
        volume: Input volume
        modality: Imaging modality

    Returns:
        Dictionary mapping window names to windowed volumes
    """
    windows = get_available_windows(modality)
    results = {}

    for window in windows:
        results[window] = apply_windowing(volume, window, modality)

    return results


def get_available_windows(modality: str) -> list:
    """
    Get list of available windows for a given modality.

    Args:
        modality: Imaging modality ('CT', 'XR', 'MG', etc.)

    Returns:
        List of available window names
    """
    # Curated MR set
    if modality == "MR":
        return ["high_contrast", "minmax", "znorm"]

    windows = []

    # Add anatomical windows if available
    if modality in ANATOMICAL_WINDOWS:
        windows.extend(ANATOMICAL_WINDOWS[modality].keys())

    # Add percentile windows if available
    if modality in PERCENTILE_WINDOWS:
        windows.extend(PERCENTILE_WINDOWS[modality].keys())

    # Always include minmax
    windows.append("minmax")

    # Include znorm for appropriate modalities
    if modality in ["MR", "XR", "MG"]:
        windows.append("znorm")

    return windows
