"""Thread-based parallel processing for I/O-bound tasks."""

import logging
from typing import List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


def thread_parallel_map(
    process_func: callable,
    tasks: List[Any],
    max_workers: int = 8,
    show_progress: bool = True,
    desc: str = "Processing",
) -> List[Any]:
    """
    Thread-based parallel map for I/O-bound tasks.

    Args:
        process_func: Function to apply to each task
        tasks: List of tasks
        max_workers: Maximum number of threads (default 8 for I/O tasks)
        show_progress: Whether to show progress bar
        desc: Description for progress bar

    Returns:
        List of results (None for failed tasks)
    """
    results = [None] * len(tasks)

    if show_progress:
        pbar = tqdm(total=len(tasks), desc=desc)

    failed_count = 0

    # Use ThreadPoolExecutor for I/O-bound tasks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_func, task): idx for idx, task in enumerate(tasks)
        }

        # Process results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]

            try:
                result = future.result()
                results[idx] = result

            except Exception as e:
                logger.error(f"Task {idx} failed: {e}")
                results[idx] = None
                failed_count += 1

            if show_progress:
                pbar.update(1)

    if show_progress:
        pbar.close()

    if failed_count > 0:
        logger.warning(f"{failed_count}/{len(tasks)} tasks failed")

    return results
