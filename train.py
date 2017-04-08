# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os

import chainer
import numpy

import icon_generator

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--outdirname", required=True)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--pretrained_srmodel")
parser.add_argument("--batchsize", type=int, default=10)
parser.add_argument("--train_sr", action="store_true")
parser.add_argument("--cropsize", type=int, default=96)
parser.add_argument('--scale', type=int, choices=[2, 4, 8, 16, 32], default=4)
parser.add_argument("--limited", action="store_true")
args = parser.parse_args()

OUTPUT_DIRECTORY = args.outdirname
os.makedirs(OUTPUT_DIRECTORY)

logging.basicConfig(filename=os.path.join(OUTPUT_DIRECTORY, "log.txt"), level=logging.DEBUG)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

logging.info(args)

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy

# paths = glob.glob("{}/*.JPEG".format(args.dataset))
paths = glob.glob(args.dataset)
dataset = icon_generator.dataset.PreprocessedImageDataset(paths=paths, cropsize=args.cropsize, resize=(300, 300), scaling_ratio=args.scale)

iterator = chainer.iterators.MultiprocessIterator(dataset, batch_size=args.batchsize, repeat=True, shuffle=True)
# iterator = chainer.iterators.SerialIterator(dataset, batch_size=args.batchsize, repeat=True, shuffle=True)

super_resolution_model = icon_generator.models.SRGenerator(times=int(numpy.log2(args.scale)))
if args.pretrained_srmodel is not None:
    chainer.serializers.load_npz(args.pretrained_srmodel, super_resolution_model)
if args.limited:
    iconizer = icon_generator.models.IconizerLimited(times=int(numpy.log2(args.scale)))
else:
    iconizer = icon_generator.models.Iconizer(times=int(numpy.log2(args.scale)))
if args.gpu >= 0:
    iconizer.to_gpu()
    super_resolution_model.to_gpu()

optimizer_iconizer = chainer.optimizers.Adam()
optimizer_iconizer.setup(iconizer)
optimizer_sr = None
if args.train_sr:
    optimizer_sr = chainer.optimizers.Adam()
    optimizer_sr.setup(super_resolution_model)

count_processed = 0
sum_loss = 0
for zipped_batch in iterator:
    low_res = chainer.Variable(xp.array([zipped[0] for zipped in zipped_batch]))
    high_res = chainer.Variable(xp.array([zipped[1] for zipped in zipped_batch]))

    iconized = iconizer(high_res)
    reconstructed = super_resolution_model(iconized)

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

    optimizer_iconizer.zero_grads()
    if optimizer_sr is not None: optimizer_sr.zero_grads()
    loss.backward()
    optimizer_iconizer.update()
    if optimizer_sr is not None: optimizer_sr.update()

    sum_loss += chainer.cuda.to_cpu(loss.data)
    report_span = args.batchsize * 100
    save_span = args.batchsize * 2000
    count_processed += high_res.shape[0]
    if count_processed % report_span == 0:
        logging.info("processed: {}".format(count_processed))
        logging.info("loss: {}".format(sum_loss / report_span))
        sum_loss = 0
    if count_processed % save_span == 0:
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "iconizer_model_{}.npz".format(count_processed)), iconizer)
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "superresolution_model_{}.npz".format(count_processed)), super_resolution_model)
