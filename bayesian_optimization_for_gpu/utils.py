import torch


def get_device():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s).")
        device = torch.device("cuda")
        x = torch.ones(1, device=device)
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"Tensor on {device}: {x}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS device found and built.")
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print(f"Tensor on {device}: {x}")
    else:
        print("No GPU devices found. Running on CPU.")
        device = torch.device("cpu")
        x = torch.ones(1, device=device)
        print(f"Tensor on {device}: {x}")
    return device


def get_supported_dtype():
    device = get_device()
    if device.type == "cuda":
        return torch.float64
    elif device.type == "mps":
        return torch.float32
    else:
        return torch.float64
