import torch


def get_device():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s).")
        device = torch.device("cuda")
        print_cuda_device_name(device)
        return device

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS device found and built.")
        device = torch.device("mps")
        print(f"Running on {device.type.upper()} device")
        return device

    else:
        print("No GPU devices found. Running on CPU.")
        device = torch.device("cpu")
        print_cuda_device_name(device)
        return device


def get_supported_dtype(device: torch.device):
    if device.type == "cuda":
        print("Using float64 for GPU computation.")
        return torch.float64
    elif device.type == "mps":
        print("Using float32 for MPS computation.")
        return torch.float32
    else:
        print("Using float64 for CPU computation.")
        return torch.float64


def print_cuda_device_name(device):
    print(f"Running on {device.type.upper()} device: {torch.cuda.get_device_name(device)}")
