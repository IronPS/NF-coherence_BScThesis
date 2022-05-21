from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

def build_horizontal_image_sequence(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    return new_im

def build_vertical_image_sequence(images):
    widths, heights = zip(*(i.size for i in images))
    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new('RGB', (max_width, total_height))
    y_offset = 0 #total_height
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    return new_im

def show_encoded_channels(img, per_channel_min_max=False):
    matplotlib.interactive(False)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3*4, 4))
    
    xs = np.arange(0, img.shape[-2])
    ys = np.arange(0, img.shape[-1])
    X, Y = np.meshgrid(xs, ys)
    
    if not per_channel_min_max:
        min_, max_ = img.min(), img.max()
        
    for i, c in enumerate(img):
        if per_channel_min_max:
            min_, max_ = c.min(), c.max()
        
        _ = axes[i].contourf(X, np.flip(Y), c, vmin=min_, vmax=max_, levels=256, cmap='gray')
        
    matplotlib.interactive(True)

    return fig

def figure_to_pil_image(fig):
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    return Image.open(buffer);

def add_text_to_PIL_image(text, orig_img, max_text_height=45):
    """
    Receives a PIL image and attaches the given text to the bottom of it
    """
    W_img, H_img = orig_img.size
                    
    H_text = max_text_height
    text_img = Image.new('RGBA', (W_img, H_text), 'white')
    draw = ImageDraw.Draw(text_img)
    w_text, h_text = draw.textsize(text)
    draw.text(( (W_img - w_text)/2, (H_text - h_text)/2 ), text, fill='black')

    return build_vertical_image_sequence([orig_img, text_img])
