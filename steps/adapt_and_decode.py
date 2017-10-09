#!/usr/bin/env python

##  Copyright (C) 2016 D S Pavan Kumar
##  dspavankumar [at] gmail [dot] com
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.


import sys
import numpy
import keras
import kaldi_io
from signal import signal, SIGPIPE, SIG_DFL
from learning_to_adapt.model import FeatureTransform, LHUC

if __name__ == '__main__':
    adaptation_feats = sys.argv[1]
    adaptation_pdfs = sys.argv[2]
    adaptation_frames = sys.argv[3]
    model = sys.argv[4]
    priors = sys.argv[5]

    if not model.endswith('.h5'):
        raise TypeError ('Unsupported model type. Please use h5 format. Update Keras if needed')

    ## Load model
    m = keras.models.load_model(model, custom_objects={'FeatureTransform': FeatureTransform, 'LHUC': LHUC})
    p = numpy.genfromtxt(priors, delimiter=',')

    with kaldi_io.SequentialBaseFloatMatrixReader("ark:-") as arkIn, \
            kaldi_io.BaseFloatMatrixWriter("ark,t:-") as arkOut:
        signal(SIGPIPE, SIG_DFL)

        for uttId, featMat in arkIn:
            logProbMat = numpy.log(m.predict(featMat) / p)
            logProbMat[logProbMat == -numpy.inf] = -100
            arkOut.write(uttId, logProbMat)
