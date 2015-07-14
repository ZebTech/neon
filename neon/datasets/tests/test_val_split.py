#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import ceil
from nose.plugins.attrib import attr

from neon.datasets.synthetic import UniformRandom, ToyImages
from neon.backends.cpu import CPU
from neon.backends.par import NoPar


class TestValidationUniformRandom(object):

    @attr('slow')
    def test_split(self):
        split = 10
        batch_size = 10
        ntrain, ntest, nin, nout = 100, 10, 10, 5
        data = UniformRandom(ntrain, ntest, nin, nout, validation_pct=split)
        data.backend = CPU(rng_seed=0)
        data.backend.actual_batch_size = batch_size
        par = NoPar()
        par.associate(data.backend)
        data.load()
        split /= 100.0
        nb_batches = ntrain // batch_size
        expected_nb_train = ceil((1.0 - split) * nb_batches)
        expected_nb_valid = nb_batches - expected_nb_train
        assert expected_nb_train == len(data.inputs['train'])
        assert expected_nb_train == len(data.targets['train'])
        assert expected_nb_valid == len(data.inputs['validation'])
        assert expected_nb_valid == len(data.targets['validation'])

    def test_round_split(self):
        split = 10
        batch_size = 32
        ntrain, ntest, nin, nout = 100, 10, 10, 5
        data = UniformRandom(ntrain, ntest, nin, nout, validation_pct=split)
        data.backend = CPU(rng_seed=0)
        data.backend.actual_batch_size = batch_size
        par = NoPar()
        par.associate(data.backend)
        data.load()
        split /= 100.0
        nb_batches = ntrain // batch_size
        expected_nb_train = ceil((1.0 - split) * nb_batches)
        expected_nb_valid = nb_batches - expected_nb_train
        assert expected_nb_train == len(data.inputs['train'])
        assert expected_nb_train == len(data.targets['train'])
        assert expected_nb_valid == len(data.inputs['validation'])
        assert expected_nb_valid == len(data.targets['validation'])


class TestValidationToyImages(object):

    @attr('slow')
    def test_split(self):
        split = 10
        batch_size = 32
        data = ToyImages(validation_pct=split)
        data.backend = CPU(rng_seed=0)
        data.backend.actual_batch_size = batch_size
        par = NoPar()
        par.associate(data.backend)
        data.load()
        ntrain = sum(a.shape[1] for a in data.inputs['train'])
        split /= 100.0
        nb_batches = ntrain // batch_size
        expected_nb_train = ceil((1.0 - split) * nb_batches)
        expected_nb_valid = nb_batches - expected_nb_train
        assert expected_nb_train == len(data.inputs['train'])
        assert expected_nb_train == len(data.targets['train'])
        assert expected_nb_valid == len(data.inputs['validation'])
        assert expected_nb_valid == len(data.targets['validation'])

    def test_round_split(self):
        split = 10
        batch_size = 30
        data = ToyImages(validation_pct=split)
        data.backend = CPU(rng_seed=0)
        data.backend.actual_batch_size = batch_size
        par = NoPar()
        par.associate(data.backend)
        data.load()
        ntrain = sum(a.shape[1] for a in data.inputs['train'])
        split /= 100.0
        nb_batches = ntrain // batch_size
        expected_nb_train = ceil((1.0 - split) * nb_batches)
        expected_nb_valid = nb_batches - expected_nb_train
        assert expected_nb_train == len(data.inputs['train'])
        assert expected_nb_train == len(data.targets['train'])
        assert expected_nb_valid == len(data.inputs['validation'])
        assert expected_nb_valid == len(data.targets['validation'])

if __name__ == '__main__':
    test = TestValidationToyImages()
    test.test_split()
    test.test_round_split()
