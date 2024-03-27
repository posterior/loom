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

import itertools
import os
import numpy
import pymetis
import pymetis._internal  # HACK to avoid errors finding .so files in path
from collections import defaultdict
from collections import namedtuple
from distributions.io.stream import json_dump
from distributions.io.stream import open_compressed
from loom.schema_pb2 import CrossCat
from loom.cFormat import assignment_stream_load
from loom.util import LoomError
from loom.util import parallel_map
import loom.store

METIS_ARGS_TEMPFILE = 'temp.metis_args.json'

Row = namedtuple('Row', ['row_id', 'group_id', 'confidence'])


def collate(pairs):
    groups = defaultdict(lambda: [])
    for key, value in pairs:
        groups[key].append(value)
    return groups.values()


def group(root, feature_name, parallel=False):
    paths = loom.store.get_paths(root, sample_count=None)
    map_ = parallel_map if parallel else itertools.starmap
    groupings = map_(
        group_sample, [(sample, feature_name) for sample in paths['samples']]
    )
    return group_reduce(groupings)


def group_sample(sample, featureid):
    model = CrossCat()
    with open_compressed(sample['model'], mode='r') as f:
        model.ParseFromString(f.read())
    for kindid, kind in enumerate(model.kinds):
        if featureid in kind.featureids:
            break
    fname = sample['assign'].encode('utf-8')
    assignments = assignment_stream_load(fname)
    return collate((a.groupids(kindid), a.rowid) for a in assignments)


def group_reduce(groupings):
    return find_consensus_grouping(groupings)


def find_consensus_grouping(groupings, debug=False):
    '''
    This implements Strehl et al's Meta-Clustering Algorithm [1].

    Inputs:
        groupings - a list of lists of lists of object ids, for example

            [
                [                   # sample 0
                    [0, 1, 2],      # sample 0, group 0
                    [3, 4],         # sample 0, group 1
                    [5]             # sample 0, group 2
                ],
                [                   # sample 1
                    [0, 1],         # sample 1, group 0
                    [2, 3, 4, 5]    # sample 1, group 1
                ]
            ]

    Returns:
        a list of Row instances sorted by (- row.group_id, row.confidence)

    References:
    [1] Alexander Strehl, Joydeep Ghosh, Claire Cardie (2002)
        "Cluster Ensembles - A Knowledge Reuse Framework
        for Combining Multiple Partitions"
        Journal of Machine Learning Research
        http://jmlr.csail.mit.edu/papers/volume3/strehl02a/strehl02a.pdf
    '''
    if not groupings:
        raise LoomError('tried to find consensus among zero groupings')

    # ------------------------------------------------------------------------
    # Set up consensus grouping problem

    groupings_list = list(groupings)
    allgroups = sum((list(g) for g in groupings_list), [])
    objects = list(set(sum(allgroups, [])))
    objects.sort()
    index = {item: i for i, item in enumerate(objects)}

    vertices = [
        numpy.array(list(map(index.__getitem__, g)), dtype=numpy.intp)
        for g in allgroups
    ]

    contains = numpy.zeros((len(vertices), len(objects)), dtype=numpy.float32)
    for v, vertex in enumerate(vertices):
      contains[v, vertex] = 1  # i.e. for u in vertex: contains[v, u] = i

    # We use the binary Jaccard measure for similarity
    overlap = numpy.dot(contains, contains.T)
    diag = overlap.diagonal()
    denom = (
        diag.reshape(len(vertices), 1) + diag.reshape(1, len(vertices)) - overlap
    )
    similarity = overlap / denom

    # ------------------------------------------------------------------------
    # Format for metis

    if not (similarity.max() <= 1):
        raise LoomError('similarity.max() = {}'.format(similarity.max()))
    similarity *= 2**16  # metis segfaults if this is too large
    int_similarity = numpy.zeros(similarity.shape, dtype=numpy.int32)
    numpy.rint(similarity, out=int_similarity, casting='unsafe')

    edges = int_similarity.nonzero()
    edge_weights = list(map(int, int_similarity[edges]))
    edges = numpy.transpose(edges)

    adjacency = [[] for _ in vertices]
    for i, j in edges:
        adjacency[i].append(int(j))

    # FIXME is there a better way to choose the final group count?
    group_count = int(numpy.median(list(map(len, groupings_list))))

    metis_args = {
        'nparts': group_count,
        'adjacency': adjacency,
        'eweights': edge_weights,
    }

    if debug:
        json_dump(metis_args, METIS_ARGS_TEMPFILE, indent=4)

    edge_cut, partition = pymetis.part_graph(**metis_args)

    if debug:
        os.remove(METIS_ARGS_TEMPFILE)

    # ------------------------------------------------------------------------
    # Clean up solution

    parts = range(group_count)
    if len(partition) != len(vertices):
        raise LoomError('metis output vector has wrong length')

    represents = numpy.zeros((len(parts), len(vertices)))
    for v, p in enumerate(partition):
        represents[p, v] = 1

    contains = numpy.dot(represents, contains)
    represent_counts = represents.sum(axis=1)
    represent_counts[numpy.where(represent_counts == 0)] = 1  # avoid NANs
    contains /= represent_counts.reshape(group_count, 1)

    bestmatch = contains.argmax(axis=0)
    confidence = contains[bestmatch, range(len(bestmatch))]
    if not all(numpy.isfinite(confidence)):
        raise LoomError('confidence is nan')

    nonempty_groups = list(set(bestmatch))
    nonempty_groups.sort()
    reindex = {j: i for i, j in enumerate(nonempty_groups)}

    grouping = [
        Row(row_id=objects[i], group_id=reindex[g], confidence=c)
        for i, (g, c) in enumerate(zip(bestmatch, confidence))
    ]

    groups = list(collate((row.group_id, row) for row in grouping))
    groups.sort(key=len, reverse=True)
    grouping = [
        Row(row_id=row.row_id, group_id=group_id, confidence=row.confidence)
        for group_id, group in enumerate(groups)
        for row in group
    ]
    grouping.sort(key=lambda x: (x.group_id, -x.confidence, x.row_id))

    return grouping
