# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os

import chainer
import numpy

import icon_generator
import cv2


def clip_img(x):
    return numpy.uint8(0 if x < 0 else (255 if x > 255 else x))
    # return numpy.float32(-1 if x < -1 else (1 if x > 1 else x))


def variable2img(x):
    print(x.data.max())
    print(x.data.min())
    img = (numpy.vectorize(clip_img)(x.data[0, :, :, :])).transpose(1, 2, 0)
    return img


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--pretrained_srmodel", required=True)
parser.add_argument("--pretrained_iconizemodel", required=True)
parser.add_argument("--batchsize", type=int, default=10)
parser.add_argument("--train_sr", action="store_true")
parser.add_argument("--cropsize", type=int, default=96)
parser.add_argument('--scale', type=int, choices=[2, 4, 8, 16, 32], default=4)
parser.add_argument("--limited", action="store_true")
args = parser.parse_args()

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy

logging.basicConfig(level=logging.INFO)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

# paths = glob.glob("{}/*.JPEG".format(args.dataset))
paths = glob.glob(args.dataset)
dataset = icon_generator.dataset.PreprocessedImageDataset(paths=paths, cropsize=args.cropsize, resize=(300, 300),
                                                          scaling_ratio=args.scale)

iterator = chainer.iterators.MultiprocessIterator(dataset, batch_size=args.batchsize, repeat=True, shuffle=True)
# iterator = chainer.iterators.SerialIterator(dataset, batch_size=args.batchsize, repeat=True, shuffle=True)

super_resolution_model = icon_generator.models.SRGenerator(times=int(numpy.log2(args.scale)))
if args.limited:
    iconizer = icon_generator.models.IconizerLimited(times=int(numpy.log2(args.scale)))
else:
    iconizer = icon_generator.models.Iconizer(times=int(numpy.log2(args.scale)))
if args.gpu >= 0:
    iconizer.to_gpu()
    super_resolution_model.to_gpu()
logging.info("load")
chainer.serializers.load_npz(args.pretrained_srmodel, super_resolution_model)
chainer.serializers.load_npz(args.pretrained_iconizemodel, iconizer)

count_processed = 0
sum_loss = 0
logging.info("loop start")
for zipped_batch in iterator:
    low_res = chainer.Variable(xp.array([zipped[0] for zipped in zipped_batch]))
    high_res = chainer.Variable(xp.array([zipped[1] for zipped in zipped_batch]))

    iconized = iconizer(high_res)
    reconstructed = super_resolution_model(iconized)

    superresolutioned = super_resolution_model(low_res)

    cv2.imwrite("check_low.png", variable2img(low_res)[:, :, ::-1])
    cv2.imwrite("check_in.png", variable2img(high_res)[:, :, ::-1])
    cv2.imwrite("check_sr.png", variable2img(superresolutioned)[:, :, ::-1])
    cv2.imwrite("check_icon.png", variable2img(iconized)[:, :, ::-1])
    cv2.imwrite("check_out.png", variable2img(reconstructed)[:, :, ::-1])

    # reconstruction loss
    loss = chainer.functions.mean_squared_error(
        high_res,
        reconstructed
    )

    # superresolution loss
    if args.train_sr:
        loss += chainer.functions.mean_squared_error(
            high_res,
            super_resolution_model(low_res)
        )

    sum_loss += chainer.cuda.to_cpu(loss.data)
    report_span = args.batchsize
    count_processed += high_res.shape[0]
    if count_processed % report_span == 0:
        logging.info("processed: {}".format(count_processed))
        logging.info("loss: {}".format(sum_loss / report_span))
        sum_loss = 0
