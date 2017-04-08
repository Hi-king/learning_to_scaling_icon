# -*- coding: utf-8 -*-
import functools
import chainer
import chainer.links.caffe
import math


def pixel_shuffle_upscale(x: chainer.Variable):
    def channel_to_axis(x: chainer.Variable, axis):
        channels = chainer.functions.separate(x, axis=1)
        result_channel = int(len(channels) / 2)
        w1, w2 = chainer.functions.stack(channels[:result_channel], axis=1), chainer.functions.stack(
            channels[result_channel:], axis=1)
        odds, evens = chainer.functions.separate(w1, axis=axis), chainer.functions.separate(w2, axis=axis)
        width_widened = chainer.functions.stack(
            functools.reduce(lambda x, y: x + y, ([a, b] for a, b in zip(odds, evens)))
            , axis=axis)
        return width_widened

    return channel_to_axis(channel_to_axis(x, 2), 3)


class SRGeneratorResBlock(chainer.Chain):
    def __init__(self):
        super().__init__(
            c1=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(64 * 3 * 3)),
            bn1=chainer.links.BatchNormalization(64),
            c2=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(64 * 3 * 3)),
            bn2=chainer.links.BatchNormalization(64),
        )

    def __call__(self, x: chainer.Variable, test=False):
        h = chainer.functions.relu(self.bn1(self.c1(x), test=test))
        h = self.bn2(self.c2(h))
        return h + x  # residual


class SRGeneratorUpScaleBlock(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv=chainer.functions.Convolution2D(in_channels=64, out_channels=256, ksize=3, stride=1, pad=1,
                                                 wscale=0.02 * math.sqrt(64 * 3 * 3))
        )

    def __call__(self, x: chainer.Variable):
        h = self.conv(x)
        h = pixel_shuffle_upscale(h)
        h = chainer.functions.relu(h)
        return h


class IconizerOld(chainer.Chain):
    def __init__(self):
        super().__init__(
            first=chainer.links.Convolution2D(3, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(3 * 3 * 3)),
            res1=SRGeneratorResBlock(),
            res2=SRGeneratorResBlock(),
            res3=SRGeneratorResBlock(),
            res4=SRGeneratorResBlock(),
            res5=SRGeneratorResBlock(),
            conv_mid=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(64 * 3 * 3)),
            bn_mid=chainer.links.BatchNormalization(64),

            downscale1=chainer.links.Convolution2D(64, 64, ksize=3, stride=2, pad=1,
                                                   wscale=0.02 * math.sqrt(64 * 3 * 3)),
            downscale2=chainer.links.Convolution2D(64, 64, ksize=3, stride=2, pad=1,
                                                   wscale=0.02 * math.sqrt(64 * 3 * 3)),

            conv_output=chainer.links.Convolution2D(64, 3, ksize=3, stride=1, pad=1,
                                                    wscale=0.02 * math.sqrt(64 * 3 * 3))
        )

    def __call__(self, x: chainer.Variable, test=False):
        h = first = chainer.functions.relu(self.first(x))

        h = self.res1(h, test=test)
        h = self.res2(h, test=test)
        h = self.res3(h, test=test)
        h = self.res4(h, test=test)
        h = self.res5(h, test=test)
        mid = self.bn_mid(self.conv_mid(h), test=test)
        h = first + mid
        h = chainer.functions.relu(self.downscale2(chainer.functions.relu(self.downscale1(h))))
        h = self.conv_output(h)
        return h


class Iconizer(chainer.Chain):
    def __init__(self, times):
        super().__init__(
            first=chainer.links.Convolution2D(3, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(3 * 3 * 3)),
            res1=SRGeneratorResBlock(),
            res2=SRGeneratorResBlock(),
            res3=SRGeneratorResBlock(),
            res4=SRGeneratorResBlock(),
            res5=SRGeneratorResBlock(),
            conv_mid=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(64 * 3 * 3)),
            bn_mid=chainer.links.BatchNormalization(64),
            downscale=chainer.ChainList(*[
                chainer.links.Convolution2D(64, 64, ksize=3, stride=2, pad=1, wscale=0.02 * math.sqrt(64 * 3 * 3)) for _
                in range(times)
                ]),
            conv_output=chainer.links.Convolution2D(64, 3, ksize=3, stride=1, pad=1,
                                                    wscale=0.02 * math.sqrt(64 * 3 * 3))
        )

    def __call__(self, x: chainer.Variable, test=False):
        h = first = chainer.functions.relu(self.first(x))

        h = self.res1(h, test=test)
        h = self.res2(h, test=test)
        h = self.res3(h, test=test)
        h = self.res4(h, test=test)
        h = self.res5(h, test=test)
        mid = self.bn_mid(self.conv_mid(h), test=test)

        h = first + mid

        for downscale_layer in self.downscale.children():
            h = chainer.functions.relu(downscale_layer(h))

        h = self.conv_output(h)
        return h


class IconizerLimited(Iconizer):
    def __init__(self, times):
        super().__init__(times)

    def __call__(self, *args, **kwargs):
        h = super().__call__(*args, **kwargs)
        h = chainer.functions.sigmoid(h)
        return h * 256


class SRGenerator(chainer.Chain):
    def __init__(self, times: int):
        super().__init__(
            first=chainer.links.Convolution2D(3, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(3 * 3 * 3)),
            res1=SRGeneratorResBlock(),
            res2=SRGeneratorResBlock(),
            res3=SRGeneratorResBlock(),
            res4=SRGeneratorResBlock(),
            res5=SRGeneratorResBlock(),
            conv_mid=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(64 * 3 * 3)),
            bn_mid=chainer.links.BatchNormalization(64),
            upscale=chainer.ChainList(
                *[SRGeneratorUpScaleBlock() for _ in range(times)]
            ),
            conv_output=chainer.links.Convolution2D(64, 3, ksize=3, stride=1, pad=1,
                                                    wscale=0.02 * math.sqrt(64 * 3 * 3))
        )

    def __call__(self, x: chainer.Variable, test=False):
        h = first = chainer.functions.relu(self.first(x))

        h = self.res1(h, test=test)
        h = self.res2(h, test=test)
        h = self.res3(h, test=test)
        h = self.res4(h, test=test)
        h = self.res5(h, test=test)
        mid = self.bn_mid(self.conv_mid(h), test=test)

        h = first + mid

        for layer in self.upscale.children():
            h = layer(h)
        h = self.conv_output(h)
        return h


class SRGeneratorOld(chainer.Chain):
    def __init__(self):
        super().__init__(
            first=chainer.links.Convolution2D(3, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(3 * 3 * 3)),
            res1=SRGeneratorResBlock(),
            res2=SRGeneratorResBlock(),
            res3=SRGeneratorResBlock(),
            res4=SRGeneratorResBlock(),
            res5=SRGeneratorResBlock(),
            conv_mid=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(64 * 3 * 3)),
            bn_mid=chainer.links.BatchNormalization(64),
            upscale1=SRGeneratorUpScaleBlock(),
            upscale2=SRGeneratorUpScaleBlock(),
            conv_output=chainer.links.Convolution2D(64, 3, ksize=3, stride=1, pad=1,
                                                    wscale=0.02 * math.sqrt(64 * 3 * 3))
        )

    def __call__(self, x: chainer.Variable, test=False):
        h = first = chainer.functions.relu(self.first(x))

        h = self.res1(h, test=test)
        h = self.res2(h, test=test)
        h = self.res3(h, test=test)
        h = self.res4(h, test=test)
        h = self.res5(h, test=test)
        mid = self.bn_mid(self.conv_mid(h), test=test)

        h = first + mid

        h = self.upscale2(self.upscale1(h))

        h = self.conv_output(h)
        return h
