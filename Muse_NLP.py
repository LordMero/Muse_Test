
import pdb

# import minimum modules
import numpy as np
import sys
import time
import getopt

# import functions and data structs
from math import log, exp
from collections import defaultdict


def ngrams_create(inp_str, n=3):
    # this function create ngrams for each string  passed in inp_str

        # we want to enhance the strin with start and stop char
        inp_str_enh = (n-1) * ["*"]
        inp_str_enh.extend(inp_str)
        inp_str_enh.append("#")
        ngrams = np.asarray([tuple(inp_str_enh[i:i+n]) for i in xrange(len(inp_str_enh)-n+1)])
        # return np array
        return ngrams




def corpus_parser(corpus_file, comment_str=";;;",dic_flag=None):

    # this function is ment to parse the file and return words or sentences in a clean format

    # dic_flag  = None means training with senetences, otherwise dic_flag =1, dictionary in the form "word phonema"

    # define lenght of comments char
    com_char = len(comment_str)
    # inizialize output
    out = []

    # read corpus line by line
    l = corpus_file.readline()
    while l:
        line = l.strip()
        # check - out for comments and skip those lines
        if (line[0:com_char] !=comment_str) and line:

            fields = line.lower().split(" ",line.count(line) if dic_flag is None else 1)
            #fields.remove("")
            # building the list to be converted in np array
            out.append(fields)
            # move to the next line
            l = corpus_file.readline()
        else:
            l = corpus_file.readline()
            # return np array
    return np.asarray(out)



def ngrams_counter(input_array, n=3):
    # this function counts the ngrams available in the entire corpus!

    # input array is of dimention 1 containing words or sentences (aka ws)
    assert input_array.ndim == 1, "Expencting an array of dim 1 contining words or sentences."
    # this bit can be improved further by using numpy structure array! But for now one dic for each ngram
    #ngram_counts = (n)*[{}]
    ngram_counts = [defaultdict(int) for i in xrange(n)]
    # loop through each ws

    # setup toolbar
    sys.stdout.write("[Training: %s]" % (" " * input_array.size))
    sys.stdout.flush()
    sys.stdout.write("\b" * (input_array.size+1)) # return to start of line, after '['


    for words in input_array:

        #time.sleep(0.1) # do real work here
        # update the bar
        sys.stdout.write("=")
        sys.stdout.flush()

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

def estimate_probs(input_str, ngram_counts, n=3):
    # this function estimates the ngrams conditional probabilities approximated with frequencies contained in ngram_counts

    # make sting comparable
    input_str = input_str.lower()
    # first get the number of unique letters or word in corpus
    tot_unique = sum(ngram_counts[0].values())
    # this function is expecting a list with n default dictionaries
    ngram_array = ngrams_create(input_str)

    # alloc conditional probabilities and weights, firs col ws, second col cond prob third col lambdas
    # note that this can be further improved by computing ex ante the maximum size for words
    cond_prob = np.array([(item, 1, 1) for item in ngram_array[:,-1]], \
                          dtype=[('ws','S50'),('logprob','f4'),('lambda','f4')])

    # set up a counter
    i = 0
    for ngram in ngram_array:
        # if my ngram is not in the hash table I'll handle the exception by giving zero to the event - this must be address by smoothing the probabilities i.e. evaluate the model, reduce other count and increase the missing ones
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
            cond_prob['logprob'][i] = log(10**-20)

        i += 1

    word_prob = np.dot(cond_prob['logprob'],cond_prob['lambda'])

    return input_str,exp(word_prob),cond_prob

def query_system(tquery_str,ngram_counts,top,n=3):

    # this function query the system to guess the top x possibility

    # defining query symbols
    q_str  = '?'
    q_word = '?*'

    # handle request
    query_str = np.chararray(1,len(tquery_str))
    query_str[:] = tquery_str

    # get index of missing ws
    inx = query_str.rindex(q_str)[0]

    # first get monograms
    monograms = ngram_counts[0];

    # allocate output
    guess_probs = np.array([(query_str.replace(q_str,mono)[0], 0) for mono in monograms.keys()],dtype=[('guessed_word','S50'),('guessed_probs','f4')])

    # compute probabilities
    for i in xrange(0,len(guess_probs['guessed_word'])):
        guess_probs['guessed_probs'][i] = estimate_probs(guess_probs['guessed_word'][i],ngram_counts,n)[1]

    # hold on to positive probabilities
    guess_probs = guess_probs[guess_probs['guessed_probs']>0]

    if guess_probs.size:
        # sort for the most likely
        guess_probs = np.sort(guess_probs,order='guessed_probs')

        for i  in xrange(1,top+1):
            return guess_probs[-i]
    else:
        return None


def main(argv):

    main_inputs = parse_options(argv)

    if len(main_inputs[0]) == 0:
        input_ws = raw_input("Please insert word to predict: ")

    pdb.set_trace()

    input_array = (corpus_parser(open(main_inputs[2],'r')))

    count_array = ngrams_counter(input_array[:,0])

    out = query_system(input_ws,count_array,main_inputs[1])

    print  out

def usage():
    print 'how to use'

def parse_options(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:t:f:n:",["input=","top=","file=","ngrams="])
    except:
        usage()
        sys.exit(2)

    if len(opts) != 0:
        for opt, arg in opts:
            if opt == '-h':
                usage()
                sys.exit()
            elif opt in ("-i", "--input"):
                 input_ws = arg
            elif opt in ("-t", "--top"):
                 top = arg
            elif opt in ("-f","--file"):
                input_file = arg
            elif opt in ("-n","--ngrams"):
                n = arg
    else:
        input_ws = ""
        top = 1
        input_file = 'cmudict-0.7b.txt'
        n = 3

    # if input_file is not provided use the default one. It's assumed to be in the same directory where the script is
    return input_ws,top,input_file,n


if __name__ == "__main__":
    main(sys.argv[1:])





