import dino.vision_transformer as vits
import torch


def dino_model_in(patch_size, vit_arch, pretrained_weights):
    # ❌ 移除 argparse

    dino = vits.__dict__[vit_arch](
        patch_size=patch_size,
        num_classes=0
    )

    state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)

    state_dict = state_dict["teacher"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    dino.load_state_dict(state_dict, strict=False)
    print(f"Pretrained weights found at {pretrained_weights}")

    return dino

def load_dino_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    patch_size = 16
    vit_arch = "vit_large"
    
    model = dino_model_in(
        patch_size = patch_size,
        vit_arch = vit_arch,
        pretrained_weights = model_path
    )

    model = model.to(device)
    model.eval()
    
    return model
