from torchvision import transforms

def CenterCropResize(crop_size=148, new_size=[64, 64], interpolation_fn=transforms.InterpolationMode.LANCZOS):
    return transforms.Compose([ \
        transforms.CenterCrop(crop_size), \
        transforms.Resize(new_size, interpolation=interpolation_fn) \
    ])

