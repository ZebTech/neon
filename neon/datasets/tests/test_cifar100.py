#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from nose.plugins.attrib import attr
from math import floor

from neon.datasets.cifar100 import CIFAR100
from neon.backends.cpu import CPU
from neon.backends.par import NoPar


class TestCIFAR100(object):

    tmp_repo = os.path.join(os.path.dirname(__file__), 'repo')

    def setup(self):
        os.makedirs(self.tmp_repo)

    def teardown(self):
        shutil.rmtree(self.tmp_repo, ignore_errors=True)

    @attr('slow')
    def test_fine_labels(self):
        data = CIFAR100(coarse=False, repo_path=self.tmp_repo)
        data.backend = CPU(rng_seed=0)
        data.backend.actual_batch_size = 128
        par = NoPar()
        par.associate(data.backend)
        data.load()
        assert len(data.inputs['train']) == 50000
        assert len(data.targets['train'][0]) == 100

    @attr('slow')
    def test_coarse_labels(self):
        data = CIFAR100(coarse=True, repo_path=self.tmp_repo)
        data.backend = CPU(rng_seed=0)
        data.backend.actual_batch_size = 128
        par = NoPar()
        par.associate(data.backend)
        data.load()
        assert len(data.inputs['train']) == 50000
        assert len(data.targets['train'][0]) == 20

    @attr('slow')
    def test_split_set_default(self):
        split = 0.2
        data = CIFAR100(coarse=False, repo_path=self.tmp_repo)
        data.backend = CPU(rng_seed=0)
        data.backend.actual_batch_size = 128
        par = NoPar()
        par.associate(data.backend)
        data.load()
        expected_valid_split = floor(len(data.input['train']) * split)
        expected_train_split = len(data.inputs['train']) - expected_valid_split
        data.split_set(split)
        assert len(data.inputs['train']) == expected_train_split
        assert len(data.inputs['validation']) == expected_valid_split

    @attr('slow')
    def test_split_set_params(self):
        split = 0.2
        to_set = 'to_set'
        data = CIFAR100(coarse=False, repo_path=self.tmp_repo)
        data.backend = CPU(rng_seed=0)
        data.backend.actual_batch_size = 128
        par = NoPar()
        par.associate(data.backend)
        data.load()
        expected_valid_split = floor(len(data.input['train']) * split)
        expected_train_split = len(data.inputs['train']) - expected_valid_split
        data.split_set(split, to_set=to_set, from_set='train')
        assert len(data.inputs['train']) == expected_train_split
        assert len(data.inputs[to_set]) == expected_valid_split
