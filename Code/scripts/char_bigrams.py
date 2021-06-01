import matplotlib.pylab as plt
from argparse import ArgumentParser
from itertools import product
from collections import defaultdict


def get_data(bible_path):
    with open(bible_path, 'r', encoding = 'utf-8') as f:
        bible_text = f.read().lower().split()
    return bible_text

def get_unique_chars(bible_text):
    unique_chars = set()
    for w in bible_text:
        for c in w:
            unique_chars.add(c)
    
    return unique_chars


def get_all_pairs(unique_chars):
    pairs = product(unique_chars, repeat = 2)
    return list(pairs)


def get_counts(bible_text, n):
    
    prefixes = defaultdict(int)
    suffixes = defaultdict(int)
    for i in range(2,n+1):
        for w in bible_text:
            if len(w)/2 >= i:
                prefixes[w[:i]] = prefixes[w[:i]] + 1
                suffixes[w[-i:]] = suffixes[w[-i:]] + 1

    return prefixes, suffixes


if __name__ == '__main__':
    parser = ArgumentParser(description = 'Get prefix and suffix bigram counts')
    parser.add_argument('bible_path', help = 'Path to bible text')
    parser.add_argument('n', help = 'n-gram value')
    args = parser.parse_args()
    bible_text = get_data(args.bible_path)
    prefixes, suffixes = get_counts(bible_text, int(args.n))

    #prefixes_to_plot = prefixes[:10]
    #suffixes_to_plot = suffixes[:10]

    pre_items = sorted(prefixes.items(), key = lambda x : x[1], reverse = True)
    suff_items = sorted(suffixes.items(), key = lambda x : x[1], reverse = True)

    pre_x, pre_y = zip(*pre_items)
    suff_x, suff_y = zip(*suff_items)

    ratio = sum(pre_y[:10])/sum(suff_y[:10])
    print(ratio)

    """
Uncomment to see graphs


    #print(pre_x[:10], pre_y[:10])
    #print(suff_x[:10], suff_y[:10])
    
    fig_p, ax_p = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax_p.plot(pre_x[:10], pre_y[:10] , color="black", label = str('Prefix Frequency'))
    #ax.legend(loc="lower right", title="",fontsize=16)
    ax_p.set_xlabel("Prefix", fontsize=16)
    ax_p.set_ylabel("Frequency", fontsize=16)

    fig_s, ax_s = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax_s.plot(suff_x[:10], suff_y[:10] , color="blue", label = str('Suffix Frequency'))
    #ax.legend(loc="lower right", title="",fontsize=16)
    ax_s.set_xlabel("Suffix", fontsize=16)
    ax_s.set_ylabel("Frequency", fontsize=16)
    plt.show()
    """
