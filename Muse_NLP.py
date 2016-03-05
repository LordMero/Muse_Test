
# let's test
import numpy as np


def ngrams_create(inp_str, n=3):
        add_bound = (n-1) * ["<s>"]
        add_bound.extend(inp_str)
        add_bound.append("<\s>")
        ngrams = np.asarray([tuple(add_bound[i:i+n]) for i in xrange(len(add_bound)-n+1)])
        return ngrams


def corpus_parser(corpus_file, comment_str=";;;"):

    # define lenght of comments char
    com_char = len(comment_str)
    out = []

    # read corpus line by line
    l = corpus_file.readline()
    while l:
        line = l.strip()
        # check - out for comments and skip those lines
        if (line[0:com_char] !=comment_str) and line:

            fields = line.split(" ")
            fields.remove("")

            out.append(fields)

            #print line
            #print fields
            #print out
            l = corpus_file.readline()
        else:
            l = corpus_file.readline()

    return np.asarray(out)

def ngrams_counter(input_array, n=3):
    # input array contains words in the first column
    ngram_counts = (n)*[{}]  # one dic for each ngram

    for words in input_array[0]:
        # create ngrams
        ngrams_array = ngrams_create(words,n)
        for ngram in ngrams_array:
            # sanity check
            assert len(ngram) == n, "Something went wrong with the size"

            for i in xrange(2, n+1):                                     #Count NE-tag 2-grams..n-grams
                ngram_counts[i-1][ngram[-i:]] += 1







