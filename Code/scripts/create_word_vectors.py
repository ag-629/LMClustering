#!/usr/bin/python3

import fasttext, re, sys
from argparse import ArgumentParser


def create_vectors(bible, lang, model, lr, dim, ws, epoch, minCount, loss):

    embedding = fasttext.train_unsupervised(bible, model = model, lr = lr, dim = dim, ws = ws, epoch = epoch, minCount = minCount, loss = loss)
    #Writes to 'embeddings' directory one elvel above
    embedding.save_model('../embeddings/'+args.lang+'.bin')

if __name__ == '__main__':
    parser = ArgumentParser(description = 'Create fasttext word vectors')
    parser.add_argument('bible', help = 'Path to tokenized Bible')
    parser.add_argument('lang', help = 'language')
    #parser.add_argument('lower', help = 'Lower case', action = 'store_true')
    parser.add_argument('--model', help = 'cbow or skipgram', default = 'skipgram', type = str)
    parser.add_argument('--lr', help = 'learning rate', default = 0.05, type = float)
    parser.add_argument('--dim', help = 'size of word vector', default = 100, type = int)
    parser.add_argument('--ws', help = 'context window size', default = 5, type = int)
    parser.add_argument('--epoch', help = 'number of epochs', default = 5, type = int)
    
    #default uses all words. This affect the clustering time greatly
    parser.add_argument('--minCount', help = 'word count minimum', default = 1, type = int)
    parser.add_argument('--loss', help = 'loss function (ns, hs, softmax)', default = 'softmax', type = str)
    args = parser.parse_args()
    

    
    with open(args.bible) as f:
        if True:
            txt = f.read().lower()
            new_bible = open('../bibles/'+args.lang+'.bible.txt', 'w', encoding = 'utf-8')
            new_bible.write(txt)
            new_bible.close()
            create_vectors('../bibles/'+args.lang+'.bible.txt', args.lang, args.model, args.lr, args.dim, args.ws, args.epoch, args.minCount, args.loss)
            
        else:
            txt = f.read()
            create_vectors('./'+args.lang+'.bible.txt', args.model, args.lr, args.dim, args.ws, args.epoch, args.minCount, args.loss)



'''
train_unsupervised parameters

    input             # training file path (required)
    model             # unsupervised fasttext model {cbow, skipgram} [skipgram]
    lr                # learning rate [0.05]
    dim               # size of word vectors [100]
    ws                # size of the context window [5]
    epoch             # number of epochs [5]
    minCount          # minimal number of word occurences [5]
    minn              # min length of char ngram [3]
    maxn              # max length of char ngram [6]
    neg               # number of negatives sampled [5]
    wordNgrams        # max length of word ngram [1]
    loss              # loss function {ns, hs, softmax, ova} [ns]
    bucket            # number of buckets [2000000]
    thread            # number of threads [number of cpus]
    lrUpdateRate      # change the rate of updates for the learning rate [100]
    t                 # sampling threshold [0.0001]
    verbose           # verbose [2]
'''

