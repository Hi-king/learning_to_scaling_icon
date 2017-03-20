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
parser.add_argument("--pretrained_srmodel", required=True)
parser.add_argument("--batchsize", type=int, default=10)
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
dataset = icon_generator.dataset.PreprocessedImageDataset(paths=paths, cropsize=96, resize=(300, 300))

iterator = chainer.iterators.MultiprocessIterator(dataset, batch_size=args.batchsize, repeat=True, shuffle=True)
# iterator = chainer.iterators.SerialIterator(dataset, batch_size=args.batchsize, repeat=True, shuffle=True)

super_resolution_model = icon_generator.models.SRGenerator()
chainer.serializers.load_npz(args.pretrained_srmodel, super_resolution_model)
if args.gpu >= 0:
    super_resolution_model.to_gpu()

iconizer = icon_generator.models.Iconizer()
optimizer = chainer.optimizers.Adam()
optimizer.setup(iconizer)

count_processed = 0
sum_loss = 0
for batch in iterator:
    data = chainer.Variable(xp.array(batch))

    iconized = iconizer(batch)
    reconstructed = super_resolution_model(iconized)

    loss = chainer.functions.mean_squared_error(
        data,
        reconstructed
    )

    sum_loss += chainer.cuda.to_cpu(loss.data)
    report_span = args.batchsize * 10
    save_span = args.batchsize * 1000
    count_processed += len(data)
    if count_processed % report_span == 0:
        logging.info("processed: {}".format(count_processed))
        logging.info("loss: {}".format(sum_loss / report_span))
        sum_loss = 0
    if count_processed % save_span == 0:
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "iconizer_model_{}.npz".format(count_processed)), iconizer)
