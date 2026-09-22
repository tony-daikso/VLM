from collections import OrderedDict
import torch
from torch import inf
import os
import torch.distributed as dist


def prefix_dict(d, prefix):
    r = OrderedDict()
    for k, v in d.items():
        r[prefix + k] = v
    return r


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def _gather_object_concat(obj):
    """Gather arbitrary python objects across ranks and concatenate lists.
    If obj is a list/tuple, returns a single flat list.
    Otherwise, returns a list of gathered objects (one per rank).
    """
    if not (dist.is_available() and dist.is_initialized()):
        # Single-process fallback
        return obj if isinstance(obj, (list, tuple)) else [obj]
    world_size = dist.get_world_size()
    gathered_list = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_list, obj)
    if isinstance(obj, (list, tuple)):
        merged = []
        for o in gathered_list:
            if isinstance(o, (list, tuple)):
                merged.extend(list(o))
            else:
                merged.append(o)
        return merged
    return gathered_list


def gather_predictions_dict(predictions):
    """
    Gathers predictions from all processes in a distributed setting.
    Supports values that are:
      - torch.Tensor: concatenated along dim=0
      - list/tuple of python scalars/strings: concatenated (preserving per-sample order)
      - str/int/float: gathered into a list
    """
    # Convert to OrderedDict and sort keys to ensure consistent ordering
    # This should not be needed, but it is to make sure the order is right and no crazy things happen!
    predictions = OrderedDict(sorted(predictions.items()))

    torch.distributed.barrier()  # Synchronize all processes before gathering
    gathered_preds = OrderedDict()

    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            # Single tensor: Gather and concatenate
            gathered_preds[k] = concat_all_gather(v)
        elif isinstance(v, (list, tuple)):
            gathered_preds[k] = _gather_object_concat(list(v))
        elif isinstance(v, (str, int, float)):
            # Gather scalars/strings into a list
            gathered_preds[k] = _gather_object_concat(v)
        else:
            raise ValueError(f"Unsupported value type for key={k}: {type(v)}")

    return gathered_preds


def gather_step_outputs(outputs):
    if len(outputs) == 0:
        return {}
    output_dict = OrderedDict()
    if isinstance(outputs[-1], list):
        outputs = outputs[0]

    for k in outputs[-1].keys():
        if k == "logs":
            output_dict[k] = gather_step_outputs([output[k] for output in outputs])
        elif isinstance(outputs[-1][k], torch.Tensor) and len(outputs[-1][k].shape) == 0:
            output_dict[k] = torch.stack([output[k] for output in outputs if k in output])
        elif isinstance(outputs[-1][k], torch.Tensor):
            output_dict[k] = torch.cat([output[k] for output in outputs if k in output], dim=0)
        elif isinstance(outputs[-1][k], list):
            merged = []
            for output in outputs:
                if k in output:
                    v = output[k]
                    if isinstance(v, list):
                        merged.extend(v)
                    else:
                        merged.append(v)
            output_dict[k] = merged
        elif isinstance(outputs[-1][k], (str, int, float)):
            # Accumulate scalars/strings as a list across steps
            output_dict[k] = [output[k] for output in outputs if k in output]
        else:
            raise ValueError(f"Unsupported value type for key={k}: {type(outputs[-1][k])}")
    return output_dict


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )
    return total_norm
