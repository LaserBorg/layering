'''
USAGE:
python layering.py --image images/image.png --mask images/mask.png --overlay images/overlay.png --name result
'''

import argparse
import cv2
import os
import sys
import numpy as np


def match_size(img1, img2):
    # make sure both images have the same size
    if img1.shape[:2] != img2.shape[:2]:
        print("Warning: Images do not have the same size.")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return img2

def ensure_alpha(img):
    image = img.copy()
    if img.shape[2] == 3:
        image = np.dstack((image, np.ones(image.shape[:2], dtype=np.uint8) * 255))
    return image


def mask_image(img, msk):
    # add alpha channel if necessary
    image = ensure_alpha(img)
    
    # also convert mask to float
    image_alpha = image[:,:,3].astype(float) / 255
    mask = msk.astype(float) / 255

    # multiply the alpha channel of the image with the mask and scale the values back up to [0, 255]
    multiplied_alpha = (image_alpha * mask * 255).astype('uint8')

    # copy the result back into the alpha channel of the image
    image[:,:,3] = multiplied_alpha

    return image


def compose_images(background, foreground):
    image = ensure_alpha(background)

    alpha_bg = image[:,:,3] / 255.0
    alpha_fg = foreground[:,:,3] / 255.0

    # combine the alpha channels
    alpha_combined = (1 - (1 - alpha_fg) * (1 - alpha_bg))

    for color in range(0, 3):
        image[:,:,color] = alpha_fg * foreground[:,:,color] + alpha_bg * background[:,:,color] * (1 - alpha_fg)

    # scale the combined alpha back up to [0, 255] and convert back to 8-bit format
    image[:,:,3] = (alpha_combined * 255).astype('uint8')

    return image
    

def main(args):
    # check if paths exist
    if not os.path.exists(args.image) or (args.mask and not os.path.exists(args.mask)) or (args.overlay and not os.path.exists(args.overlay)):
        raise FileNotFoundError("One of the paths does not exist.")

    # load the image unchanged (RGB / RGBA) and the mask as grayscale
    image = cv2.imread(args.image, -1)


    # if mask is provided, mask the image
    if args.mask:
        mask = cv2.imread(args.mask, 0)
        # make sure image and mask have the same size
        mask = match_size(image, mask)
        # mask image by multiplying the alpha channel with the mask
        image = mask_image(image, mask)


    # if overlay is provided, compose it above the image
    if args.overlay:
        overlay = cv2.imread(args.overlay, -1)

        # check if overlay has an alpha channel
        if overlay.shape[2] == 4:
            # make sure image and overlay have the same size
            overlay = match_size(image, overlay)
            # compose overlay over image
            image = compose_images(image, overlay)
        else:
            print("Warning: Overlay requires alpha channel.")


    # save the resulting image
    output_filename = os.path.splitext(args.image)[0] + "_" + args.name + ".png"
    cv2.imwrite(output_filename, image, [cv2.IMWRITE_PNG_COMPRESSION, args.compression])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments, so run the test code
        cmd = 'python layering.py --image images/image.png --mask images/mask.png --overlay images/overlay.png --name result'
        os.system(cmd)

    else:
        # Arguments provided, so parse them and run the main function
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True, help="path to the image (RGB or RGBA)")
        ap.add_argument("-m", "--mask", required=False, help="path to the input mask")
        ap.add_argument("-o", "--overlay", required=False, help="path to the overlay (RGBA)")
        ap.add_argument("-n", "--name", required=False, default="result", help="naming extension for the output image")
        ap.add_argument("-c", "--compression", required=False, default=9, help="png compression level")
        args = ap.parse_args()

        ## alternatively, convert the argparse.Namespace object to a dictionary for easier access
        # args = vars(ap.parse_args())
        ## then call it with keywords as argument
        # image_path = args["image"]

        main(args)
