# -*- coding: utf-8 -*-
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--scale", required=True, type=int)
args = parser.parse_args()

input_img = cv2.imread(args.input)
assert input_img.shape[0] % args.scale == 0
assert input_img.shape[1] % args.scale == 0
output_img = cv2.resize(input_img,
                        (
                            input_img.shape[0] // args.scale,
                            input_img.shape[1] // args.scale,
                        )
                        )
cv2.imwrite(args.output, output_img)
