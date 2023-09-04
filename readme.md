# Layering Tool

simple command line script in python that supports 2 operations via parameter:
- mask an image (RGB / RGBA) with another image ("mask", RGB or grayscale)
- compose another image (RGBA "overlay") on top

currently supports only 8bit color space.

### Usage:
> python layering.py --image images/image.png --mask images/mask.png --overlay images/overlay.png --name result

### Example:

Image:  
<img src="images/image.png" width="200" height="200">

Mask:  
<img src="images/mask.png" width="200" height="200">

Overlay:  
<img src="images/overlay.png" width="200" height="200">

Result:  
<img src="images/image_result.png" width="200" height="200">
