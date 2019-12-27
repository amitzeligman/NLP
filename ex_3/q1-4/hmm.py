import os
import time
from data import *
from collections import defaultdict, Counter

def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print("Start training")
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = [defaultdict(lambda: defaultdict(int)) for i in range(5)]
    ### YOUR CODE HERE
    q_tri_list = []
    q_bi_list = []
    q_uni_list = []
    e_word_tag_list = []
    e_tag_list = []
    for sent in sents:
        words, tags = zip(*sent)
        prev_prev_tag = '*'
        prev_tag = '*'
        for word, tag in zip(words, tags):

            total_tokens += 1
            q_tri_list.append((tag, prev_prev_tag, prev_tag))
            q_bi_list.append((tag, prev_tag))
            q_uni_list.append(tag)

            e_word_tag_list.append((word, tag))
            e_tag_list.append(tag)

            prev_prev_tag = prev_tag
            prev_tag = tag
        q_tri_list.append(('STOP', prev_prev_tag, prev_tag))
        q_bi_list.append(('STOP', prev_tag))
        q_bi_list.append(('*', '*'))
        q_uni_list.append('STOP')
        q_uni_list.append('*')
        q_uni_list.append('*')
        # Add 'STOP', '*', '*'
        total_tokens += 3

    for comb in q_tri_list:
        if not comb in q_tri_counts.keys():
            q_tri_counts[comb] = 1
        else:
            q_tri_counts[comb] = q_tri_counts[comb] + 1
    for comb in q_bi_list:
        if not comb in q_bi_counts.keys():
            q_bi_counts[comb] = 1
        else:
            q_bi_counts[comb] = q_bi_counts[comb] + 1
    for comb in q_uni_list:
        if not comb in q_uni_counts.keys():
            q_uni_counts[comb] = 1
        else:
            q_uni_counts[comb] = q_uni_counts[comb] + 1

    for comb in e_word_tag_list:
        if not comb in e_word_tag_counts.keys():
            e_word_tag_counts[comb] = 1
        else:
            e_word_tag_counts[comb] = e_word_tag_counts[comb] + 1

    for comb in e_tag_list:
        if not comb in e_tag_counts.keys():
            e_tag_counts[comb] = 1
        else:
            e_tag_counts[comb] = e_tag_counts[comb] + 1

    for key in q_tri_counts:
        q_tri_counts[key] = q_tri_counts[key] / q_bi_counts[key[1:][::-1]]

    for key in q_bi_counts:
        q_bi_counts[key] = q_bi_counts[key] / q_uni_counts[key[-1]]

    for key in e_word_tag_counts:
        e_word_tag_counts[key] = e_word_tag_counts[key] / q_uni_counts[key[-1]]

    for key in q_uni_counts:
        q_uni_counts[key] = q_uni_counts[key] / total_tokens

    ### YOUR CODE HERE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                e_word_tag_counts, e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE

    n = len(sent)

    # Tag pruning
    thresh = 1e-2
    possible_tags = [tag for tag in e_tag_counts.keys()]
    possible_tags_pairs = []
    for tag1 in possible_tags:
        for tag2 in possible_tags:
            if (tag1, tag2) in q_bi_counts.keys() and q_bi_counts[(tag1, tag2)] > thresh:
                possible_tags_pairs.append((tag1, tag2))

    def transmission_prob(tag, prev_prev_tag, prev_tag):
        tri = q_tri_counts[(tag, prev_prev_tag, prev_tag)] if (tag, prev_prev_tag, prev_tag) in q_tri_counts.keys() else 0
        bi = q_bi_counts[(tag, prev_tag)] if (tag, prev_tag) in q_bi_counts.keys() else 0
        uni = q_uni_counts[tag] if tag in q_uni_counts.keys() else 0

        return lambda1 * tri + lambda2 * bi + (1 - lambda1 - lambda2) * uni

    def find_max(word, k, u, v, pai_prev):
        max_ = 0
        argmax_ = 'O'
        emission = e_word_tag_counts[(word, v)] if (word, v) in e_word_tag_counts.keys() else 0
        if k == 0 or k == 1:
            pai_new = (pai_prev[('*', u)] if k else 1) * transmission_prob(v, '*', u) * emission
            return pai_new, '*'

        for w in possible_tags:
            if (u, w) in possible_tags_pairs:
                pai_new = (pai_prev[(w, u)] if k else 1) * transmission_prob(v, w, u) * emission
                if pai_new > max_:
                    max_ = pai_new
                    argmax_ = w
        return max_, argmax_

    pai = [] * n
    bp = [] * n
    for k, word in enumerate(sent):
        pai_k = dict()
        bp_k = dict()

        if not k:
            for v in possible_tags:
                pai_k[('*', v)], bp_k[('*', v)] = find_max(word, k, '*', v, pai[k-1] if k else 1)
                for u in possible_tags:
                    if (v, u) in possible_tags_pairs:
                        pai_k[(u, v)], bp_k[(u, v)] = 0, '*'

        else:
            for v, u in possible_tags_pairs:
                pai_k[(u, v)], bp_k[(u, v)] = find_max(word, k, u, v, pai[k-1] if k else 1)
        pai.append(pai_k)
        bp.append(bp_k)

    # Backward calculation of tags
    argmax = ('O', 'O')
    max_ = 0
    for v, u in possible_tags_pairs:
        if pai[n - 1][(u, v)] * transmission_prob('STOP', u, v) > max_:
            max_ = pai[n - 1][(u, v)] * transmission_prob('STOP', u, v)
            argmax = (u, v)

    predicted_tags[-2:] = argmax

    for k in reversed(range(n - 2)):
        predicted_tags[k] = bp[k+2][predicted_tags[k+1], predicted_tags[k+2]]

    ### YOUR CODE HERE
    return predicted_tags


def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print("Start evaluation")
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        pred_tag_seqs.append(hmm_viterbi(words, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                                         e_tag_counts, 0.4, 0.4))
        ### YOUR CODE HERE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)


if __name__ == "__main__":
    start_time = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)

    hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
             e_word_tag_counts, e_tag_counts)

    train_dev_end_time = time.time()
    print("Train and dev evaluation elapsed: " + str(train_dev_end_time - start_time) + " seconds")

