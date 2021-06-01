import re
from fasttext import load_model
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser(description = 'Convert .bin to .vec format')
    parser.add_argument('bin_file', help = 'Path to .bin file from Fasttext embedding')
    parser.add_argument('lang', help = 'Language')
    args = parser.parse_args()
    # original BIN model loading
    f = load_model(args.bin_file)

    lines=[]

    # get all words from model
    words = f.get_words()
    #lang = re.match(r'[^.]+', args.bin_file).group()
    with open('./'+args.lang+'.vec','w') as file_out:
    
        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(f.get_dimension()) + "\n")

        # line by line, you append vectors to VEC file
        for w in words:
            v = f.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr+'\n')
            except:
                pass
