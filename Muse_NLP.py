
from pdb import set_trace as bpoint

# ======================================================================
#                        import modules
# ======================================================================
import numpy as np
import sys
import getopt
import string

# import functions and data structs
from math import log, exp
from collections import defaultdict








# ======================================================================
#                        Define Main Functions
# ======================================================================


def corpus_parser(corpus_file, comment_str,sent_splitter, dic_flag):

    # this function is ment to parse the file and return words or sentences in a clean format

    # dic_flag  = None means training with senetences, otherwise dic_flag =1, dictionary in the form "word phonema"

    # define lenght of comments char
    com_char = len(comment_str)
    # rem punctuation
    exclude = set(string.punctuation)
    exclude.remove(".")

    # inizialize output
    out = []
    # read corpus line by line
    l = corpus_file.readline()
    while l:
        line = l.strip()
        # check - out for comments and skip those lines
        if (line[0:com_char] !=comment_str) and line:

            if  dic_flag is None:
                line   = ''.join(ch for ch in line if ch not in exclude)

                # if I have text type trayining file
                fields = line.lower().split(sent_splitter)[0]
                fields = [eol+" " for eol in fields.split(" ") if eol !=""]
                bpoint
            else:
                # if I have a dictionary style training file I don't want other symbols in my training sample.
                fool    = line.lower().split(" ",1)
                fields  = list(fool[0])

            # building the list to be converted in np array
            out.append(fields)
            # move to the next line
            l = corpus_file.readline()
        else:
            l = corpus_file.readline()
            # return np array

    return np.asarray(out)


def ngrams_create(inp_str, n):
    # this function create ngrams for each string  passed in inp_str

        # we want to enhance the strin with start and stop char

        inp_str_enh = (n-1) * ["*"]
        inp_str_enh.extend([items for items in inp_str])
        inp_str_enh.append("#")

        ngrams = ([tuple(inp_str_enh[i:i+n]) for i in xrange(len(inp_str_enh)-n+1)])
        # return np array
        return np.asarray(ngrams)




def ngrams_counter(input_array, n):
    # this function counts the ngrams available in the entire corpus!

    # input array is of dimention 1 containing words or sentences (aka ws)
    assert input_array.ndim == 1, "Expencting an array of dim 1 contining words or sentences."
    # this bit can be improved further by using numpy structure array! But for now one dic for each ngram
    #ngram_counts = (n)*[{}]
    ngram_counts = [defaultdict(int) for i in xrange(n)]
    # loop through each ws

    for words in input_array:

        # create ngrams for each ws
        ngrams_array = ngrams_create(words,n)
        for ngram in ngrams_array:
            # sanity check about dimentions
            assert len(ngram) == n, "Something went wrong with the size"
            # Count 2-grams..n-grams
            for i in xrange(2, n+1):
                ngram_counts[i-1][''.join(ngram[-i:])] += 1
            if ngram[-1] != "#":
                ngram_counts[0][ngram[-1:][0]] += 1

    sys.stdout.write("\n")

    return ngram_counts

def estimate_probs(tinput_str, ngram_counts, n):
    # this function estimates the ngrams conditional probabilities approximated with frequencies contained in ngram_counts

    # make sting comparable and put into a list as ngrams_create is expecting that
    if tinput_str.index(" "):
        # need to preserve spaces
        input_str  = [eol+" " for eol in tinput_str.split(" ") if eol !=""]

    else:
        input_str = list(tinput_str)
    # first get the number of unique letters or word in corpus
    tot_unique = sum(ngram_counts[0].values())
    # this function is expecting a list with n default dictionaries
    ngram_array = ngrams_create(input_str,n)

    # alloc conditional probabilities and weights, firs col ws, second col cond prob third col lambdas
    # note that this can be further improved by computing ex ante the maximum size for words

    #pdb.set_trace()
    cond_prob = np.array([(item, 1, 1) for item in ngram_array[:,-1]], \
                          dtype=[('ws','S50'),('logprob','f4'),('lambda','f4')])

    # set up a counter

    i = 0
    for ngram in ngram_array:
        # if my ngram is not in the hash table I'll handle the exception by giving zero to the event - this must be address by smoothing the probabilities i.e. evaluate the model, reduce other counts and increase the missing ones

        try:
            if ngram[n-2] == '*':
                num = ngram_counts[0][''.join(ngram[-1])]
                den = tot_unique
            else:
                num = ngram_counts[n-1][''.join(ngram[:])]
                den = ngram_counts[n-2][''.join(ngram[-n:-1])]
            # store probabilities in log format
            cond_prob['logprob'][i] = log(float(num)/den)

        except:
            cond_prob['logprob'][i] = log(10**-60)

        i += 1

    word_prob = np.dot(cond_prob['logprob'],cond_prob['lambda'])

    return input_str,exp(word_prob),cond_prob

def query_system(tquery_str,ngram_counts,top,n):

    # this function query the system to guess the top x possibilities


    # handle request
    query_str = np.chararray(1,len(tquery_str))
    query_str[:] = tquery_str

    # get index of missing ws
    try:
        inx = query_str.rindex('*')[0]
        inx = xrange(inx-1,inx)
        wcard = '?*'
    except:
        try:
            inx = query_str.rindex('?')[0]
            wcard = '?'
        except ValueError:
            print "Expecting '?' or '?*' as wildcards."
            sys.exit(2)

    # first get monograms
    monograms = ngram_counts[0];

    # allocate arrays
    guess_probs = np.array([(query_str.replace(wcard,mono)[0], 0) for mono in monograms.keys()],dtype=[('guessed_word','S50'),('guessed_probs','f4')])
    out         = np.empty((1,top),dtype='S2')

    # compute probabilities
    for i in xrange(0,len(guess_probs['guessed_word'])):
        guess_probs['guessed_probs'][i] = estimate_probs(guess_probs['guessed_word'][i],ngram_counts,n)[1]

    # hold on to positive probabilities
    guess_probs = guess_probs[guess_probs['guessed_probs']>0]

    if guess_probs.size:
        # sort for the most likely
        guess_probs = np.sort(guess_probs,order='guessed_probs')

        for i  in xrange(1,min(guess_probs.size,top+1)):
            print guess_probs['guessed_word'][-1]
            #out[0,i-1] = guess_probs['guessed_word'][-i]

        #return out
    else:
        return None






# ======================================================================
#              Define Utility & Windon Dressing Functions
# ======================================================================


def usage():
    print """
    Muse NLP usage: Muse_NLP.py [--version] [--help]
                                [-t | --top int val] [-f | --file <path>]
                                [-n | --ngrams int val]

        This script uses ngrams as Natural Language Processing model to guess missing letters or words.
        Please use '?' or '?*' respectively to indicate the expecting letter or word (i.e. 'Hel?o' of 'Hello ?*').
        Arguments:

            -i not optional, input string: -i 'Hel?o'
            -t optional number of top guesses, defualt: -t 1
            -f path to training file
            -n optional number of ngrams to be calculated, default: -n 3
    """
def version():
    print """
        Muse NLP for python 2.x, version 1.0.0
        Written by Graziano Mirata (graziano.mirata@gmail.com)
    """


class parse_inputs(object):
    """
    Parses inputs from command line
    """

    def __init__(self):
            # set defaults
        self.top         = 1
        self.input_file  = 'cmudict-0.7b.txt'
        self.n           = 3
        self.dic_flag    = 1
        self.split       = None
        self.comment_str =";;;"
        self.input_str   = ''

    def parse_options(self, argv):

        self.input_str = argv[0].lower()
        #self.n = temp(self.input_str)
        try:
            opts, args = getopt.getopt(argv[1:],"hvs:t:f:d:n:c:",\
            ["help","version","top=","dfile=","file=","split=","comm=","ngrams="])
        except getopt.GetoptError:
            usage()
            sys.exit(2)

        for opt, arg in opts:
            if opt in ("-h","--help"):
                usage()
                sys.exit()
            elif opt in ("-v","--version"):
                version()
                sys.exit()
            elif opt in ("-t", "--top"):
                self.top = int(arg)
            elif opt in ("-d","--dfile"):
                self.input_file = arg
                self.dic_flag   = 1
            elif opt in ("-f","--file"):
                self.input_file = arg
                self.dic_flag   = None
            elif opt in ("-s","--split"):
                self.split = arg
            elif opt in ("-c","--comm"):
                self.comment_str = arg
            elif opt in ("-n","--ngrams"):
                self.n = int(arg)

    # if input_file is not provided use the default one. It's assumed to be in the same directory where the script is
    #return input_ws,top,input_file,n,dic_flag,split,comment_str



def temp(input_str):
    n = input_str.index('?')
    return n




# ======================================================================
#                        MAIN
# ======================================================================


def main(argv):

    i_parser = parse_inputs()


    i_parser.parse_options(argv)

    input_array = corpus_parser(open(i_parser.input_file,'r'),i_parser.comment_str,i_parser.split, i_parser.dic_flag)

    count_array = ngrams_counter(input_array,i_parser.n)


    query_system(i_parser.input_str,count_array,i_parser.top,i_parser.n)


if __name__ == "__main__":
    main(sys.argv[1:])
