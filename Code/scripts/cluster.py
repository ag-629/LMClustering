#!/usr/bin/env python3
#coding: utf-8

#python cluster.py -C -S 0 -l --no-N --no-D --no-b -t 0.5 Maltese.vec ./UD_Maltese-MUDT-master/mt_mudt-ud-test.conllu 

#from czech_stemmer import cz_stem

import fasttext
import functools
import argparse
import sys, re
from collections import defaultdict, Counter
from sortedcollections import ValueSortedDict
from collections import OrderedDict
import time

#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import homogeneity_completeness_v_measure

from numpy import inner
from numpy.linalg import norm

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from ast import literal_eval as make_tuple
from sklearn.metrics import confusion_matrix
import itertools

from sklearn.cluster import AgglomerativeClustering

from pyjarowinkler import distance

#from new_editdistance import *
from edit_distance import *

import unidecode

from tqdm import tqdm

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging

ap = argparse.ArgumentParser(
        description='find lemma for form as nearest lemma in emb space')

ap.add_argument('embeddings', help='file with the embeddings')


#ap.add_argument('conllu_test',help='file with the forms and lemmas')
#ap.add_argument('norm', help = 'file with pre-computed normalizing factor')

ap.add_argument('confidences',
                help = 'file with string:confidence dictionary')

ap.add_argument('lang', help = 'language')

ap.add_argument('output_path', help = 'output path')

ap.add_argument("-l", "--lowercase", action="store_true",
        help="lowercase input forms")
ap.add_argument("-S", "--stems", type=int, default=2,
        help="Use stems of length S (first S characters, but see also M and D)")
ap.add_argument("-R", "--remerge", type=int,
        help="Remerge clusters on test data using stem length R")

#Tune this?
ap.add_argument("-r", "--remergethreshold", type=float, default=0.3,
        help="threshold when remerging")

ap.add_argument("-D", "--devow", action="store_true",
        help="Devowel stems")
ap.add_argument("--no-D", dest = "-D", action = "store_false")
ap.add_argument("-P", "--postags", type=str,
        help="Read in a POS tag disctionary and add POS to stems")

ap.add_argument("-n", "--number", type=int,
        help="How many embeddings to read in")
ap.add_argument("-V", "--verbose", action="store_true",
        help="Print more verbose progress info")
ap.add_argument("-N", "--normalize", action="store_true",
        help="Normalize the embeddings")
ap.add_argument("--no-N", dest = "-N", action = "store_false")
#ap.add_argument("-b", "--baselines", action="store_true",
        #help="Compute baselines and upper bounds")
#ap.add_argument("--no-b", dest = "-b", action = "store_false")

#tune this
ap.add_argument("-t", "--threshold", type=float, default=0.30,
        help="Do not perform merges with avg distance greater than this")

ap.add_argument("-O", "--oov", type=str, default="guess",
        help="OOVs: keep/guess")
# TODO unused
ap.add_argument("-p", "--plot", type=str,
        help="Plot the dendrogramme for the given stem")
ap.add_argument("-m", "--merges", action="store_true",
        help="Write out the merges")
ap.add_argument("-M", "--measure", type=str, default='average',
        help="Linkage measure average/complete/single")
#ap.add_argument("-s", "--similarity", type=str,help="Similarity: edit or cos or jw")
ap.add_argument("-C", "--clusters", action="store_true",
        help="Print out the clusters.")
ap.add_argument("-L", "--length", type=float, default=0.05,
        help="Weight for length similarity")
args = ap.parse_args()


level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=level)


# TODO how to do this right?
OOV_EMB_SIM = 0.9


# https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/plot_hierarchical_clustering_dendrogram.py
# Authors: Mathew Kallada
# License: BSD 3 clause
"""
=========================================
Plot Hierarachical Clustering Dendrogram 
=========================================
This example plots the corresponding dendrogram of a hierarchical clustering
using AgglomerativeClustering and the dendrogram method available in scipy.
"""

import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

#cosine similarity
def embsim(word, otherword):
    if word in embedding and otherword in embedding:
        emb1 = embedding[word]
        emb2 = embedding[otherword]
        #print('w1 embedding: ', emb1)
        #print('w2 embedding: ', emb2)

        sim = inner(emb1, emb2)/(norm(emb1)*norm(emb2))
        #sim = cosine_similarity([emb1], [emb2])
        #logging.debug(sim)
        assert sim >= -1.0001 and sim <= 1.0001, "Cos sim must be between -1 and 1"
        # shift to 0..1 range
        sim = (sim+1)/2
    else:
        # backoff
        sim = OOV_EMB_SIM
    return sim


def lensim(word, otherword):
    return 1 / (1 + args.length * abs(len(word) - len(otherword)) )


def similarity(word, otherword, word_conf, otherword_conf):
    #print("IN SIMILARITY FUNCTION")
    # edit_distance(str1, str2, m, n, norm, str1_conf, str2_conf)
    #print(edit_distance(word, otherword, len(word), len(otherword), norm, word_conf, otherword_conf))
    #sys.exit(0)
    #print('1 - ed: ',(1 - edit_distance(word, otherword, len(word), len(otherword), word_conf, otherword_conf)))
    #print('1 - ed * es: ',(1 - edit_distance(word, otherword, len(word), len(otherword), word_conf, otherword_conf)) * embsim(word, otherword))
    #print('#####################################################################################')
    #print('#####################################################################################')
    return (1 - edit_distance(word, otherword, len(word), len(otherword), word_conf, otherword_conf)) * embsim(word, otherword)


def get_stem(form, remerging=False):

    if args.lowercase:
        form = form.lower()

    if args.devow:
        form = devow(form)

    if remerging:
        stem = form[:args.remerge]
    else:
        stem = form[:args.stems]
    
    if args.postags:
        stem = stem + '_' + postag[form]

    return stem
    # return cz_stem(form, aggressive=False)

    
logging.info('Read in embeddings')
if args.embeddings.endswith('.bin'):
    # get word embedding still the same way, i.e. as embedding[word]
    # TODO no iterating over this
    # (or if, then iterate over embedding.words)
    embedding = fasttext.load_model(args.embeddings)
else:
    #denom = eval(open(args.norm).read())
    conf_dict = eval(open(args.confidences).read())
    embedding = defaultdict(list)
    forms_stemmed = defaultdict(set)
    form_freq_rank = dict()
    with open(args.embeddings) as embfile:
        size, dim = map(int, embfile.readline().split())
        print('Size: ', size)
        if args.number:
            size = min(size, args.number)
        for i in range(size):
            fields = embfile.readline().split()
            form = fields[0]
            #print(form)
            emb = list(map(float, fields[1:]))
            if args.normalize:
                emb /= norm(emb)
            if args.lowercase and not form.islower():
                form = form.lower()
                if form in embedding:
                    # do not overwrite "bush" with "Bush"
                    continue
            embedding[form] = emb
            stem = get_stem(form)
            forms_stemmed[stem].add(form)
            form_freq_rank[form] = i



if args.verbose:
    for form in sorted(embedding.keys()):
        logging.debug(form + ' -> ' + get_stem(form))


def get_dist(form1, form2, word_conf, otherword_conf):
    
    #kept the same name of function which now returnsd a distance
    return 1 - similarity(form1, form2, word_conf, otherword_conf)


# list of indexes -> list of words
def node2str(node, index2word):
    return [index2word[index] for index in node]


def linkage(cluster1, cluster2, D):
    linkages = list()
    for node1 in cluster1:
        for node2 in cluster2:
            linkages.append(D[node1, node2])
    # min avg max
    if args.measure == 'average':
        return sum(linkages)/len(linkages)
    elif args.measure == 'single':
        return min(linkages)
    elif args.measure == 'complete':
        return max(linkages)
    else:
        assert False


# cluster each hypercluster
logging.info('Run the main loop')
#print('1')
#iterate_over = forms_stemmed
#if args.plot:
#    iterate_over = [args.plot]


def cl(stem, cluster):
    return stem + '___' + str(cluster)


def aggclust(forms_stemmed, conf_dict):
    # form -> cluster

    result = dict()
    for stem in forms_stemmed:
        # vocabulary
        index2word = list(forms_stemmed[stem])
        I = len(index2word)
        #print('2')
        logging.debug(stem)
        logging.debug(I)
        logging.debug(index2word)
        print('Building Distance Matrix...')
        if I == 1:
            result[index2word[0]] = cl(stem, 0)
            continue

        D = np.empty((I, I))
        for i1 in tqdm(range(I)):
            if index2word[i1] in conf_dict:
                conf_word = conf_dict[index2word[i1]]
                for i2 in range(I):
                    if index2word[i2] in conf_dict:
                        conf_otherword = conf_dict[index2word[i2]]
                        D[i1,i2] = get_dist(index2word[i1], index2word[i2] , conf_word, conf_otherword)
        print('Calling sklearn Aggclustering..')
        clustering = AgglomerativeClustering(affinity='precomputed',
                                             linkage = args.measure, distance_threshold = 0.3, n_clusters=None)
        clustering.fit(D)

        print(len(set(clustering.labels_)))
        print(clustering.n_clusters_)
        sys.exit(0)
        for c in clustering.labels_:
            print(c)
        

        return clustering
    """
        print("Merging Clusters...")
        # default: each has own cluster
        clusters = list(range(I))
        nodes = [[i] for i in range(I)]
        for merge in clustering.children_:
            # check stopping criterion
            #print(merge[0],merge[1])
            #print(linkage(nodes[merge[0]], nodes[merge[1]], D))
            if args.threshold < linkage(nodes[merge[0]], nodes[merge[1]], D):
                break
            # perform the merge
            nodes.append(nodes[merge[0]] + nodes[merge[1]])
            # reassign words to new cluster ID
            for i in nodes[-1]:
                clusters[i] = len(nodes) - 1
        for i, cluster in enumerate(clusters):
            #print(index2word[i])
            result[index2word[i]] = cl(stem, cluster)
    return result
"""
                

#if args.plot:
#        plt.title('Hierarchical Clustering Dendrogram')
#        plot_dendrogram(clustering, labels=index2word)
#        plt.show()

def writeout_clusters(clustering):
    #lang = re.match(r'[^.]+', args.embeddings).group()
    out = open(args.output_path+'_'+str(args.threshold), 'w', encoding = 'utf-8')
    cluster2forms = defaultdict(list)
    for form, cluster in clustering.items():
        cluster2forms[cluster].append(form)
    for cluster in sorted(cluster2forms.keys()):
        #print('CLUSTER', cluster)
        for form in cluster2forms[cluster]:
            out.write(str(form)+'\n')
        out.write(str('\n'))
    sys.stdout.flush()
    out.close()

clusterset = set()

# each cluster name becomes its most frequent wordform
def rename_clusters(clustering):
    cluster2forms = defaultdict(list)
    for form, cluster in clustering.items():
        cluster2forms[cluster].append(form)

    cluster2newname = dict()
    for cluster, forms in cluster2forms.items():
        form2rank = dict()
        for form in forms:
            assert form in form_freq_rank
            form2rank[form] = form_freq_rank[form]
        most_frequent_form = min(form2rank, key=form2rank.get)
        cluster2newname[cluster] = most_frequent_form
        clusterset.add(most_frequent_form)

    new_clustering = dict()
    for form, cluster in clustering.items():
        new_clustering[form] = cluster2newname[cluster]

    return new_clustering

# now 1 nearest neighbour wordform;
# other option is nearest cluster in avg linkage
# (probably similar result but not necesarily)
def find_cluster_for_form(form, clustering):
    stem = get_stem(form)
    cluster = form  # backoff: new cluster
    if args.oov == "guess" and stem in forms_stemmed:
        dists = dict()
        for otherform in forms_stemmed[stem]:
            dists[otherform] = get_dist(form, otherform)
        nearest_form = min(dists, key=dists.get)
        if dists[nearest_form] < args.threshold:
            cluster = clustering[nearest_form]
            # else leave the default, i.e. a separate new cluster
    return cluster


cluster_remerged = dict()
start = time.time()

clustering = aggclust(forms_stemmed, conf_dict)
logging.info('Rename clusters')
renamed_clustering = rename_clusters(clustering)
if args.clusters:
    logging.info('Write out train clusters')
    print('START TRAIN CLUSTERS')
    writeout_clusters(renamed_clustering)
    print('END TRAIN CLUSTERS')
    #logging.info('Run evaluation')
    #hcva = homogeneity(renamed_clustering, writeout=args.clusters)
    #print('Homogeneity', 'completenss', 'vmeasure', 'accuracy', sep='\t')
    #print(*hcva, sep = '\t')

execTime = time.time() - start
print('Execution Time: ' + str(execTime))
logging.info('Done.')

