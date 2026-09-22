import logging
import os
import sys
import torch.distributed as dist


def set_loglevel(debug):
    # Set non-zero ranks to WARNING level to reduce output
    # Check both environment variable (before init) and dist.get_rank() (after init)
    rank = None
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])

    if rank is not None and rank != 0:
        loglevel = logging.WARN
    else:
        loglevel = logging.INFO if debug else logging.WARN
    logger.setLevel(loglevel)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(loglevel)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False


# Set it to debug by default. Will override in main.
logger = logging.getLogger("main")
set_loglevel(debug=True)
