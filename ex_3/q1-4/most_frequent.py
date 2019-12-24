import os
from data import *
from collections import defaultdict

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    The dictionary should have a default value.
    """
    ### YOUR CODE HERE

    model = dict()

    for sent in train_data:
        words, tags = zip(*sent)
        for word, tag in zip(words, tags):
            if word in model.keys():
                model[word].append(tag)
            else:
                model[word] = []
                model[word].append(tag)

    for word in model.keys():
        model[word] = max(set(model[word]), key=model[word].count)

    return model

    ### YOUR CODE HERE

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_set:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        pred_tag_seqs.append(tuple(map(lambda p: pred_tags[p], words)))
        ### YOUR CODE HERE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)

if __name__ == "__main__":
    os.chdir('/Users/amitzeligman/Git/NLP/ex_3/')   # TODO
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")
    os.chdir('/Users/amitzeligman/Git/NLP/ex_3/q1-4') # TODO
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    most_frequent_eval(dev_sents, model)

