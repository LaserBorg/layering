'''
USAGE:
python layering.py --image images/image.png --mask images/mask.png --overlay images/overlay.png --name result
'''

import argparse
import cv2
import os
import sys

def compose_images(background, foreground):
    result = background.copy()

    alpha_bg = background[:,:,3] / 255.0
    alpha_fg = foreground[:,:,3] / 255.0

    # combine the alpha channels
    alpha_combined = (1 - (1 - alpha_fg) * (1 - alpha_bg))

    for color in range(0, 3):
        result[:,:,color] = alpha_fg * foreground[:,:,color] + alpha_bg * background[:,:,color] * (1 - alpha_fg)

    # scale the combined alpha back up to [0, 255] and convert back to 8-bit format
    result[:,:,3] = (alpha_combined * 255).astype('uint8')

    return result
    

def mask_image(img, msk):
    image = img.copy()
    mask = msk.copy()

    # convert the alpha channel of the image and the mask to float and scale the values down to [0, 1]
    image_alpha = image[:,:,3].astype(float) / 255
    mask = mask.astype(float) / 255

    # multiply the alpha channel of the image with the mask and scale the values back up to [0, 255]
    multiplied_alpha = (image_alpha * mask * 255).astype('uint8')

    # copy the result back into the alpha channel of the image
    image[:,:,3] = multiplied_alpha

    return image


def main(args):
    # check if paths exist
    if not os.path.exists(args.image) or not os.path.exists(args.mask) or (args.overlay and not os.path.exists(args.overlay)):
        raise FileNotFoundError("One of the paths does not exist.")

    # load the image unchanged (RGBA) and the mask as grayscale
    image = cv2.imread(args.image, -1)
    mask = cv2.imread(args.mask, 0)

    # check if image and mask have the same size
    if image.shape[:2] != mask.shape[:2]:
        print("Warning: Image and mask do not have the same size.")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # mask image by multiplying the alpha channel with the mask
    image = mask_image(image, mask)
    

    # if overlay path is provided, layer overlay over the modified image
    if args.overlay:
        foreground = cv2.imread(args.overlay, -1)
        
        # make sure image and overlay have the same size
        if image.shape[:2] != foreground.shape[:2]:
            print("Warning: Image and overlay do not have the same size.")
            foreground = cv2.resize(foreground, (image.shape[1], image.shape[0]))
        
        # compose overlay over image
        image = compose_images(image, foreground)

    # save the resulting image
    output_filename = os.path.splitext(args.image)[0] + "_" + args.name + ".png"
    cv2.imwrite(output_filename, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])



if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments, so run the test code
        cmd = 'python layering.py --image images/image.png --mask images/mask.png --overlay images/overlay.png --name result'
        os.system(cmd)

    else:
        # Arguments provided, so parse them and run the main function
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True, help="path to the input image")
        ap.add_argument("-m", "--mask", required=True, help="path to the input mask")
        ap.add_argument("-o", "--overlay", required=False, help="path to the input overlay")
        ap.add_argument("-n", "--name", required=False, default="result", help="naming extension for the output image")
        args = ap.parse_args()

        ## alternatively, convert the argparse.Namespace object to a dictionary
        # args = vars(ap.parse_args())
        ## then call it with keywords as argument
        # image_path = args["image"]

        main(args)
