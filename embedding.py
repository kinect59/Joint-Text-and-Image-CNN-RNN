from __future__ import print_function
import sys
import numpy as np

def load(vocab, dimension, filename):
    print('loading embeddings from "%s"' % filename, file=sys.stderr)
    embedding = np.zeros((max(vocab.values()) + 1, dimension), dtype=np.float32)
    seen = set()
    with open(filename) as fp:
        for line in fp:
            tokens = line.strip().split(' ')
            if len(tokens) == dimension + 1:
                word = tokens[0]
                if word in vocab:
                    embedding[vocab[word]] = [float(x) for x in tokens[1:]]
                    seen.add(word)
                    if len(seen) == len(vocab):
                        break
    return embedding

if __name__ == '__main__':
    # can be used to filter an embedding file
    if len(sys.argv) != 3:
        print('usage: cat wordlist | %s <dimension> <embedding_filename>' % sys.argv[0])
        sys.exit(1)

    vocab = {word.strip(): i for i, word in enumerate(sys.stdin.readlines())}
    dimension = int(sys.argv[1])
    filename = sys.argv[2]
    embedding = load(vocab, dimension, filename)

    for word, i in vocab.items():
        print(word, ' '.join([str(x) for x in embedding[i]]))
