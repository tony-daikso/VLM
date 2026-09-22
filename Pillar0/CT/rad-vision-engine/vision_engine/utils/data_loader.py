"""
Unified data loader for Vision Engine exports.
Supports LZ4 and Video (HEVC in MKV/MP4 containers). JPEG2000 support has been removed.
"""

import torch
import numpy as np
import tarfile
import json
import io
import lz4.frame
import subprocess
import tempfile
import logging
import os
from pathlib import Path
from PIL import Image
from typing import Dict, Union, Optional, List
from contextlib import contextmanager
import nibabel as nib

logger = logging.getLogger(__name__)


@contextmanager
def _library_path_context():
    """Context manager to temporarily remove LD_LIBRARY_PATH to avoid loading libraries multiple times."""
    LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH")
    if LD_LIBRARY_PATH is not None:
        del os.environ["LD_LIBRARY_PATH"]
    try:
        yield
    finally:
        if LD_LIBRARY_PATH is not None:
            os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH


class HardwareAcceleration:
    """Check and manage hardware acceleration capabilities."""

    @staticmethod
    def has_nvidia_gpu() -> bool:
        """Check if NVIDIA GPU is available."""
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @staticmethod
    def has_nvdec() -> bool:
        """Check if NVIDIA hardware decoder is available."""
        if not HardwareAcceleration.has_nvidia_gpu():
            return False

        try:
            # Check if ffmpeg has nvdec support
            result = subprocess.run(
                ["ffmpeg", "-decoders"], capture_output=True, text=True
            )
            return "hevc_cuvid" in result.stdout
        except FileNotFoundError:
            return False

    @staticmethod
    def get_decoder_args(codec: str = "hevc") -> List[str]:
        """Get appropriate decoder arguments based on available hardware."""
        if HardwareAcceleration.has_nvdec():
            logger.debug("Using NVIDIA hardware acceleration for video decoding")
            if codec == "hevc":
                return ["-c:v", "hevc_cuvid"]

        # Fallback to software decoding
        logger.info("Using software decoding (no hardware acceleration available)")
        return []


def _load_video_volume_hw(
    tar: tarfile.TarFile, video_member: tarfile.TarInfo, metadata: Dict
) -> np.ndarray:
    """
    Load and decode video file to numpy array with hardware acceleration support.

    Args:
        tar: Open tarfile object
        video_member: TarInfo for the video file (supports .mkv and .mp4)
        metadata: Metadata dictionary with HU mapping info

    Returns:
        numpy array with restored HU values
    """
    # Extract video to temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Use the original extension from the member name
        video_ext = Path(video_member.name).suffix
        video_path = temp_path / f"volume{video_ext}"

        # Extract video file
        f = tar.extractfile(video_member)
        with open(video_path, "wb") as out:
            out.write(f.read())

        # Get video info using ffprobe
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name",
            "-of",
            "json",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"Failed to probe video: {result.stderr}")

        video_info = json.loads(result.stdout)
        stream = video_info["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        codec = stream.get("codec_name", "hevc")

        # Get hardware acceleration args
        decoder_args = HardwareAcceleration.get_decoder_args(codec)

        # Decode video to raw frames using ffmpeg with optional hardware acceleration
        cmd = ["ffmpeg"]
        if decoder_args:
            cmd.extend(decoder_args)
        cmd.extend(
            [
                "-i",
                str(video_path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray16le",  # 16-bit grayscale output
                "-",
            ]
        )

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            # Fallback to software decoding if hardware fails
            if decoder_args:
                logger.warning("Hardware decoding failed, falling back to software")
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "gray16le",
                    "-",
                ]
                result = subprocess.run(cmd, capture_output=True)

            if result.returncode != 0:
                raise ValueError(f"Failed to decode video: {result.stderr.decode()}")

        # Convert raw bytes to numpy array
        raw_data = np.frombuffer(result.stdout, dtype=np.uint16)

        # Calculate number of frames from data size
        pixels_per_frame = width * height
        total_pixels = len(raw_data)
        num_frames = total_pixels // pixels_per_frame

        if total_pixels % pixels_per_frame != 0:
            logger.warning(
                f"Raw data size {total_pixels} is not divisible by frame size {pixels_per_frame}"
            )

        volume_16bit = raw_data[: num_frames * pixels_per_frame].reshape(
            (num_frames, height, width)
        )

        # Restore HU values from 16-bit encoding
        export_info = metadata.get("export_info", {})
        hu_mapping = export_info.get("hu_mapping", {})
        hu_min = hu_mapping.get("min", -1024)
        hu_max = hu_mapping.get("max", 3071)
        hu_range = hu_max - hu_min

        # Map from 16-bit (0-65535) back to HU values
        volume = (volume_16bit.astype(np.float32) / 65535.0 * hu_range + hu_min).astype(
            np.int16
        )

        return volume


def _load_video_volume(
    tar: tarfile.TarFile, video_member: tarfile.TarInfo, metadata: Dict
) -> np.ndarray:
    """
    Load and decode video file to numpy array.

    Args:
        tar: Open tarfile object
        video_member: TarInfo for the video file (supports .mkv and .mp4)
        metadata: Metadata dictionary with HU mapping info

    Returns:
        numpy array with restored HU values
    """
    # Extract video to temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Use the original extension from the member name
        video_ext = Path(video_member.name).suffix
        video_path = temp_path / f"volume{video_ext}"

        # Extract video file
        f = tar.extractfile(video_member)
        with open(video_path, "wb") as out:
            out.write(f.read())

        # Get video info using ffprobe
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"Failed to probe video: {result.stderr}")

        video_info = json.loads(result.stdout)
        stream = video_info["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])

        # Decode video to raw frames using ffmpeg
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le",  # 16-bit grayscale output
            "-",
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise ValueError(f"Failed to decode video: {result.stderr.decode()}")

        # Convert raw bytes to numpy array
        raw_data = np.frombuffer(result.stdout, dtype=np.uint16)

        # Calculate number of frames from data size
        pixels_per_frame = width * height
        total_pixels = len(raw_data)
        num_frames = total_pixels // pixels_per_frame

        if total_pixels % pixels_per_frame != 0:
            logger.warning(
                f"Raw data size {total_pixels} is not divisible by frame size {pixels_per_frame}"
            )

        volume_16bit = raw_data[: num_frames * pixels_per_frame].reshape(
            (num_frames, height, width)
        )

        # Restore HU values from 16-bit encoding
        export_info = metadata.get("export_info", {})
        hu_mapping = export_info.get("hu_mapping", {})
        hu_min = hu_mapping.get("min", -1024)
        hu_max = hu_mapping.get("max", 3071)
        hu_range = hu_max - hu_min

        # Map from 16-bit (0-65535) back to HU values
        # Note: Video was encoded with full 16-bit range, even though it's 10-bit codec
        volume = (volume_16bit.astype(np.float32) / 65535.0 * hu_range + hu_min).astype(
            np.int16
        )

        return volume


def load_vision_sample(
    tarball_path: str,
    device: Union[str, torch.device] = "cpu",
    use_hardware_acceleration: bool = True,
) -> torch.Tensor:
    """
    Load a vision engine tarball and restore the original raw volume.

    Supports formats:
    - LZ4 compressed tarballs (.tar.lz4)
    - Video compressed tarballs (.tar with .mkv or .mp4 inside) or folders
    - Direct video files (.mkv or .mp4)

    Args:
        tarball_path: Path to the tarball file produced by vision engine, or a directory containing extracted files
        device: PyTorch device to load the tensor to ('cpu', 'cuda', etc.)
        use_hardware_acceleration: Use GPU decoding for video if available

    Returns:
        3D tensor with original values (e.g., Hounsfield Units for CT)
        Shape: (D, H, W) where D is number of slices

    Examples:
        >>> # Load any format transparently
        >>> volume = load_vision_sample('/path/to/data.tar.lz4')
        >>> volume = load_vision_sample('/path/to/data.tar.gz')
        >>> volume = load_vision_sample('/path/to/data.tar')

        >>> # Load from directory with MKV or MP4 video
        >>> volume = load_vision_sample('/path/to/extracted_dir/')  # Contains volume.mkv or volume.mp4

        >>> # Load directly to GPU
        >>> volume = load_vision_sample('/path/to/data.tar', device='cuda')

        >>> # Disable hardware acceleration
        >>> volume = load_vision_sample('/path/to/data.tar', use_hardware_acceleration=False)
    """
    volume = None
    metadata = None

    # Use context manager to avoid loading libraries multiple times
    with _library_path_context():
        # Directory support: unarchived outputs (e.g., volume.mkv + metadata.json)
        path_obj = Path(tarball_path)
        if path_obj.is_dir():
            metadata_path = path_obj / "metadata.json"
            if not metadata_path.exists():
                raise ValueError(f"No metadata.json found in directory: {tarball_path}")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Check for known content types, prefer video → npy
            mkv_path = path_obj / "volume.mkv"
            mp4_path = path_obj / "volume.mp4"
            npy_path = path_obj / "volume.npy"

            # Check for video files (mkv or mp4)
            video_path = None
            if mkv_path.exists():
                video_path = mkv_path
            elif mp4_path.exists():
                video_path = mp4_path

            if video_path:
                # Probe video
                cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,codec_name",
                    "-of",
                    "json",
                    str(video_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise ValueError(f"Failed to probe video: {result.stderr}")
                info = json.loads(result.stdout)
                stream = info["streams"][0]
                width = int(stream["width"])
                height = int(stream["height"])
                codec = stream.get("codec_name", "hevc")

                # Build ffmpeg decode command
                cmd = ["ffmpeg"]
                if use_hardware_acceleration:
                    decoder_args = HardwareAcceleration.get_decoder_args(codec)
                    if decoder_args:
                        cmd.extend(decoder_args)
                cmd.extend(
                    [
                        "-i",
                        str(video_path),
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "gray16le",
                        "-",
                    ]
                )
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0 and use_hardware_acceleration:
                    # Fallback to software decoding
                    cmd = [
                        "ffmpeg",
                        "-i",
                        str(video_path),
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "gray16le",
                        "-",
                    ]
                    result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    raise ValueError(
                        f"Failed to decode video: {result.stderr.decode()}"
                    )

                raw = np.frombuffer(result.stdout, dtype=np.uint16)
                pixels_per_frame = width * height
                total_pixels = len(raw)
                num_frames = total_pixels // pixels_per_frame
                if total_pixels % pixels_per_frame != 0:
                    logger.warning(
                        f"Raw data size {total_pixels} is not divisible by frame size {pixels_per_frame}"
                    )
                vol16 = raw[: num_frames * pixels_per_frame].reshape(
                    (num_frames, height, width)
                )

                modality = metadata["series_info"]["modality"]
                if modality == "CT":
                    # Restore HU
                    export_info = metadata.get("export_info", {})
                    hu_mapping = export_info.get("hu_mapping", {})
                    hu_min = hu_mapping.get("min", -1024)
                    hu_max = hu_mapping.get("max", 3071)
                    hu_range = hu_max - hu_min
                    volume = (
                        vol16.astype(np.float32) / 65535.0 * hu_range + hu_min
                    ).astype(np.int16)
                else:
                    volume = vol16

            elif npy_path.exists():
                volume = np.load(str(npy_path))
            else:
                raise ValueError(
                    f"No recognized data found in directory (expected volume.mkv, volume.mp4, or volume.npy): {tarball_path}"
                )

            if isinstance(volume, np.ndarray):
                volume = torch.from_numpy(volume).to(device)
            return volume

        # Determine compression type from extension (with header sniff fallback)
        if tarball_path.endswith(".tar.lz4") or tarball_path.endswith(".torch.tar"):
            compression_type = "lz4"
        elif tarball_path.endswith(".tar"):
            compression_type = None  # Expected plain tar, but we will sniff to be safe
        else:
            compression_type = None  # Unknown/other; we will sniff

        # Open tarball based on compression type, with LZ4 header sniffing for misnamed files
        if compression_type == "lz4":
            # Check if it's actually uncompressed tar
            if tarfile.is_tarfile(tarball_path):
                tar = tarfile.open(tarball_path, "r")
            else:
                with open(tarball_path, "rb") as f:
                    compressed_data = f.read()
                decompressed_data = lz4.frame.decompress(compressed_data)
                tar_buffer = io.BytesIO(decompressed_data)
                tar = tarfile.open(fileobj=tar_buffer, mode="r")
        else:
            # Either '.tar' or unknown extension. Sniff header and fallback to LZ4 if needed
            try:
                if tarfile.is_tarfile(tarball_path):
                    tar = tarfile.open(tarball_path, "r")
                else:
                    # Read header to detect LZ4 frame magic (\x04\x22\x4d\x18)
                    with open(tarball_path, "rb") as f:
                        header = f.read(4)
                        f.seek(0)
                        compressed_data = f.read()
                    if header == b"\x04\x22\x4d\x18":
                        decompressed_data = lz4.frame.decompress(compressed_data)
                        tar_buffer = io.BytesIO(decompressed_data)
                        tar = tarfile.open(fileobj=tar_buffer, mode="r")
                    else:
                        # Last resort: try LZ4 decompress; if it fails, raise
                        try:
                            decompressed_data = lz4.frame.decompress(compressed_data)
                            tar_buffer = io.BytesIO(decompressed_data)
                            tar = tarfile.open(fileobj=tar_buffer, mode="r")
                        except Exception as _e:
                            raise ValueError(
                                f"Unsupported format or corrupt file: {tarball_path}"
                            )
            except Exception:
                raise

        try:
            # Get all members
            members = tar.getmembers()

            # Check format by looking at contents
            volume_member = next((m for m in members if m.name == "volume.npy"), None)
            mkv_member = next((m for m in members if m.name == "volume.mkv"), None)
            mp4_member = next((m for m in members if m.name == "volume.mp4"), None)
            torch_member = next((m for m in members if m.name == "volume.pt"), None)

            # Use whichever video format is available
            video_member = mkv_member or mp4_member

            # Load metadata
            metadata = None
            for member in members:
                if member.name == "metadata.json":
                    f = tar.extractfile(member)
                    metadata = json.load(f)
                    break

            if metadata is None:
                raise ValueError(f"No metadata.json found in {tarball_path}")

            # Load volume based on format
            if torch_member:
                # PyTorch tensor format
                f = tar.extractfile(torch_member)
                volume = torch.load(io.BytesIO(f.read()))
                # Convert to numpy for consistent return type, will be converted back to torch at the end
                # Handle bfloat16 which doesn't have direct numpy support
                # if tensor.dtype == torch.bfloat16:
                #     volume = tensor.float().numpy()
                # else:
                #     volume = tensor.numpy()
                # logger.info(f"Loaded torch tensor with shape {volume.shape}, original dtype {tensor.dtype}")

            elif video_member:
                # Video format
                if use_hardware_acceleration:
                    volume = _load_video_volume_hw(tar, video_member, metadata)
                else:
                    volume = _load_video_volume(tar, video_member, metadata)
                logger.debug(
                    f"Loaded video volume with shape {volume.shape}, dtype {volume.dtype}"
                )

            elif volume_member:
                # Numpy volume format (LZ4)
                f = tar.extractfile(volume_member)
                volume = np.load(io.BytesIO(f.read()))
                logger.info(
                    f"Loaded numpy volume with shape {volume.shape}, dtype {volume.dtype}"
                )

            else:
                # Legacy individual numpy slices
                npy_members = sorted(
                    [
                        m
                        for m in members
                        if m.name.endswith(".npy") and m.name != "volume.npy"
                    ]
                )

                if npy_members:
                    # Individual numpy slices (legacy format)
                    slices = []
                    for member in npy_members:
                        f = tar.extractfile(member)
                        slice_data = np.load(io.BytesIO(f.read()))
                        slices.append(slice_data)

                    volume = np.stack(slices, axis=0)
                    logger.info(f"Loaded {len(npy_members)} numpy slices")

                else:
                    raise ValueError(
                        f"No recognized volume data found in {tarball_path}"
                    )

        finally:
            tar.close()

        # Convert to torch tensor and move to device
        ## if load volume is a numpy array, convert to torch tensor
        # breakpoint()
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume).to(device)

        return volume


def get_export_info(tarball_path: str) -> Dict[str, any]:
    """
    Get metadata and export information from a vision engine tarball.

    Args:
        tarball_path: Path to the tarball file

    Returns:
        Dictionary containing metadata and export information
    """
    # Open tarball based on format
    if tarball_path.endswith(".tar.lz4"):
        # LZ4 format requires decompression first
        with open(tarball_path, "rb") as f:
            compressed_data = f.read()
        decompressed_data = lz4.frame.decompress(compressed_data)
        tar_buffer = io.BytesIO(decompressed_data)
        tar = tarfile.open(fileobj=tar_buffer, mode="r")
    elif tarball_path.endswith(".tar"):
        tar = tarfile.open(tarball_path, "r")
    else:
        raise ValueError(f"Unsupported format: {tarball_path}")

    try:
        # Load metadata
        metadata_member = tar.getmember("metadata.json")
        f = tar.extractfile(metadata_member)
        metadata = json.load(f)

        # Detect format
        members = tar.getnames()
        if "volume.mkv" in members or "volume.mp4" in members:
            format_type = "video"
        elif "volume.npy" in members:
            format_type = "numpy"
        else:
            format_type = "unknown"

        metadata["format_type"] = format_type
        return metadata

    finally:
        tar.close()


def load_nifti(
    nifti_path: str, device: Union[str, torch.device] = "cpu"
) -> torch.Tensor:
    """
    Load a NIfTI file (.nii, .nii.gz) and convert to torch tensor.

    Args:
        nifti_path: Path to NIfTI file
        device: PyTorch device to load the tensor to ('cpu', 'cuda', etc.)

    Returns:
        3D or 4D tensor with shape (D, H, W) or (T, D, H, W)

    Examples:
        >>> # Load standard NIfTI
        >>> volume = load_nifti('/path/to/scan.nii.gz')

        >>> # Load directly to GPU
        >>> volume = load_nifti('/path/to/scan.nii.gz', device='cuda')
    """
    # Load NIfTI file
    nifti_img = nib.load(nifti_path)

    # Reorient to standard RAS+ orientation
    nifti_img = nib.as_closest_canonical(nifti_img)

    # Get data as numpy array
    data = nifti_img.get_fdata()

    # Convert to appropriate dtype (same as DICOM handling)
    # Infer from data range
    if data.min() < -500 and data.max() > 500:
        # Likely CT data in HU
        volume = data.astype(np.int16)
    else:
        # Keep as float for other modalities
        volume = data.astype(np.float32)

    # Convert to torch tensor
    tensor = torch.from_numpy(volume).to(device)

    logger.info(f"Loaded NIfTI volume with shape {tensor.shape}, dtype {tensor.dtype}")

    return tensor
