# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
# Copyright (c) 2015, Google, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import csv
import mock
import numpy.random
from nose.tools import raises
from distributions.io.stream import open_compressed, json_load, json_dump
from loom.util import LoomError, tempdir
from loom.test.util import get_test_kwargs, CLEANUP_ON_ERROR
import loom.store
import loom.format
import loom.datasets
import loom.tasks
import pytest

GARBAGE = 'XXX garbage XXX'


def csv_load(filename):
    with open_compressed(filename, 'rt') as f:
        reader = csv.reader(f)
        return list(reader)


def csv_dump(data, filename):
    with open_compressed(filename, 'wt') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_missing_schema_error(dataset):
    kwargs = get_test_kwargs(dataset)
    with pytest.raises(LoomError):
        with tempdir(cleanup_on_error=CLEANUP_ON_ERROR) as store:
            with mock.patch('loom.store.STORE', new=store):
                rows_csv = kwargs['rows_csv']
                schema = os.path.join(store, 'missing.schema.json')
                loom.tasks.ingest(kwargs['name'], schema, rows_csv, debug=True)


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_missing_rows_error(dataset):
    kwargs = get_test_kwargs(dataset)
    with pytest.raises(LoomError):
        with tempdir(cleanup_on_error=CLEANUP_ON_ERROR) as store:
            with mock.patch('loom.store.STORE', new=store):
                schema = kwargs['schema']
                rows_csv = os.path.join(store, 'missing.rows_csv')
                loom.tasks.ingest(kwargs['name'], schema, rows_csv, debug=True)


def _test_modify_csv(modify, dataset):
    kwargs = get_test_kwargs(dataset)
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR) as store:
        with mock.patch('loom.store.STORE', new=store):
            schema = kwargs['schema']
            encoding = kwargs['encoding']
            rows = kwargs['rows']
            rows_dir = os.path.join(store, 'rows_csv')
            loom.format.export_rows(encoding, rows, rows_dir)
            rows_csv = os.path.join(rows_dir, os.listdir(rows_dir)[0])
            data = csv_load(rows_csv)
            data = modify(data)
            csv_dump(data, rows_csv)
            loom.tasks.ingest(kwargs['name'], schema, rows_csv, debug=True)


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_csv_ok(dataset):
    modify = lambda data: data
    _test_modify_csv(modify, dataset)


def shuffle_columns(data):
    data = list(zip(*data))
    numpy.random.shuffle(data)
    data = list(zip(*data))
    return data


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_csv_shuffle_columns_ok(dataset):
    modify = shuffle_columns
    _test_modify_csv(modify, dataset)


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_csv_extra_column_ok(dataset):
    modify = lambda data: [row + [GARBAGE] for row in data]
    _test_modify_csv(modify, dataset)
    modify = lambda data: [[GARBAGE] + row for row in data]
    _test_modify_csv(modify, dataset)


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_csv_missing_column_error(dataset):
    with pytest.raises(LoomError):
        modify = lambda data: [row[:-1] for row in data]
        _test_modify_csv(modify, dataset)


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_csv_missing_header_error(dataset):
    with pytest.raises(LoomError):
        modify = lambda data: data[1:]
        _test_modify_csv(modify, dataset)


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_csv_repeated_column_error(dataset):
    with pytest.raises(LoomError):
      modify = lambda data: [row + row for row in data]
      _test_modify_csv(modify, dataset)


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_csv_garbage_header_error(dataset):
    with pytest.raises(LoomError):
        modify = lambda data: [[GARBAGE] * len(data[0])] + data[1:]
        _test_modify_csv(modify, dataset)


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_csv_short_row_error(dataset):
    with pytest.raises(LoomError):
        modify = lambda data: data + [data[1][:1]]
        _test_modify_csv(modify, dataset)


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_csv_long_row_error(dataset):
    with pytest.raises(LoomError):
        modify = lambda data: data + [data[1] + data[1]]
        _test_modify_csv(modify, dataset)


def _test_modify_schema(modify, dataset):
    kwargs = get_test_kwargs(dataset)
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR) as store:
        with mock.patch('loom.store.STORE', new=store):
            schema = kwargs['schema']
            rows_csv = kwargs['rows_csv']
            modified_schema = os.path.join(store, 'schema.json')
            data = json_load(schema)
            data = modify(data)
            json_dump(data, modified_schema)
            loom.tasks.ingest(kwargs['name'], modified_schema, rows_csv, debug=True)


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_schema_ok(dataset):
    modify = lambda data: data
    _test_modify_schema(modify, dataset)


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_schema_empty_error(dataset):
    with pytest.raises(LoomError):
        modify = lambda data: {}
        _test_modify_schema(modify, dataset)


@pytest.mark.parametrize('dataset', loom.datasets.TEST_CONFIGS)
def test_schema_unknown_model_error(dataset):
    with pytest.raises(LoomError):
        modify = lambda data: {key: GARBAGE for key in data}
        _test_modify_schema(modify, dataset)
