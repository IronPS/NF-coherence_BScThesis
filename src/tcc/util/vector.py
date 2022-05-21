import torch

def interpolate(tensor1, tensor2, num_its=7):
    assert num_its > 0

    weight_interval = 1./num_its
    interpolated_data = []
    for i in range(1, num_its + 1):
        interpolated_data.append(torch.lerp(tensor1, tensor2, weight_interval * i))

    return torch.stack(interpolated_data).squeeze()
