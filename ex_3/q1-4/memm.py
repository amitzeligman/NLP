from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import os
import numpy as np
from collections import defaultdict

def build_extra_decoding_arguments(train_sents):
    """
    Receives: all sentences from training set
    Returns: all extra arguments which your decoding procedures requires
    """

    extra_decoding_arguments = {}
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### YOUR CODE HERE

    return extra_decoding_arguments


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE
    features['next_word'] = next_word
    features['prev_word'] = prev_word
    features['prevprev_word'] = prevprev_word
    features['prev_tag'] = prev_tag
    features['prevprev_prev_tag'] = prevprev_tag + ' ' +prev_tag
    ### YOUR CODE HERE
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<st>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<st>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in range(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels


def memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE

    n = len(sent)
    for k, word in enumerate(sent):
        curr_word = sent[k][0]
        prev_word = sent[k - 1][0] if k > 0 else '<st>'
        prevprev_word = sent[k - 2][0] if k > 1 else '<st>'
        prev_tag = predicted_tags[k - 1] if k > 0 else '*'
        prevprev_tag = predicted_tags[k - 2] if k > 1 else '*'
        next_word = sent[k + 1][0] if k < (len(sent) - 1) else '</s>'

        features = extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag)
        features_vectorized = vec.transform(features)

        approx_tag = logreg.predict(features_vectorized)
        predicted_tags[k] = index_to_tag_dict[approx_tag[0]]
    ### YOUR CODE HERE
    return predicted_tags

def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    n = len(sent)

    # Tag pruning
    thresh = 1e-2
    possible_tags = [tag for tag in index_to_tag_dict.values()]
    possible_tags = possible_tags[:-1]
    possible_tags_pairs = []
    for tag1 in possible_tags:
        for tag2 in possible_tags:
            possible_tags_pairs.append((tag1, tag2))

    def log_lin_model(tag, prev_prev_tag, prev_tag, log_reg, curr_word, prev_word, prevprev_word, next_word, vec):
        features = extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prev_prev_tag)
        features_vectorized = vec.transform(features)

        probs = log_reg.predict_proba(features_vectorized)



        return probs[0][tag_to_idx_dict[tag]]

    def find_max(curr_word, k, u, v, pai_prev, prev_word, prevprev_word, next_word, vec):
        max_ = 0
        argmax_ = 'O'

        if k == 0 or k == 1:
            pai_new = (pai_prev[('*', u)] if k else 1) * log_lin_model(v, '*', u, logreg, curr_word, prev_word, prevprev_word, next_word, vec)
            return pai_new, '*'

        for w in possible_tags:
            if (u, w) in possible_tags_pairs:
                pai_new = (pai_prev[(w, u)] if k else 1) * log_lin_model(v, w, u, logreg, curr_word, prev_word, prevprev_word, next_word, vec)
                if pai_new > max_:
                    max_ = pai_new
                    argmax_ = w
        return max_, argmax_

    pai = [] * n
    bp = [] * n
    for k, word in enumerate(sent):
        pai_k = dict()
        bp_k = dict()
        prev_word = sent[k - 1][0] if k > 0 else '<st>'
        prevprev_word = sent[k - 2][0] if k > 1 else '<st>'
        next_word = sent[k + 1][0] if k < (len(sent) - 1) else '</s>'
        if not k:
            for v in possible_tags:
                pai_k[('*', v)], bp_k[('*', v)] = find_max(word, k, '*', v, pai[k - 1] if k else 1, prev_word, prevprev_word, next_word, vec)
                for u in possible_tags:
                    if (v, u) in possible_tags_pairs:
                        pai_k[(u, v)], bp_k[(u, v)] = 0, '*'

        else:
            for v, u in possible_tags_pairs:
                pai_k[(u, v)], bp_k[(u, v)] = find_max(word, k, u, v, pai[k - 1] if k else 1, prev_word, prevprev_word, next_word, vec)
        pai.append(pai_k)
        bp.append(bp_k)

    # Backward calculation of tags
    argmax = ('O', 'O')
    max_ = 0
    for v, u in possible_tags_pairs:

        if pai[n - 1][(u, v)]> max_:
            max_ = pai[n - 1][(u, v)]
            argmax = (u, v)

    predicted_tags[-2:] = argmax

    for k in reversed(range(n - 2)):
        predicted_tags[k] = bp[k + 2][predicted_tags[k + 1], predicted_tags[k + 2]]

    ### YOUR CODE HERE
    return predicted_tags

def should_log(sentence_index):
    if sentence_index > 0 and sentence_index % 10 == 0:
        if sentence_index < 150 or sentence_index % 200 == 0:
            return True

    return False


def memm_eval(test_data, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = time.time()

    correct_greedy_preds = 0
    correct_viterbi_preds = 0
    total_words_count = 0

    gold_tag_seqs = []
    greedy_pred_tag_seqs = []
    viterbi_pred_tag_seqs = []
    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        greedy_pred_tag_seqs.append(memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments))
        viterbi_pred_tag_seqs.append(memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments))
        ### YOUR CODE HERE

    greedy_evaluation = evaluate_ner(gold_tag_seqs, greedy_pred_tag_seqs)
    viterbi_evaluation = evaluate_ner(gold_tag_seqs, viterbi_pred_tag_seqs)

    return greedy_evaluation, viterbi_evaluation

def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict


if __name__ == "__main__":
    full_flow_start = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    extra_decoding_arguments = build_extra_decoding_arguments(train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    vec = DictVectorizer()
    print("Create train examples")
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)


    num_train_examples = len(train_examples)
    print("#example: " + str(num_train_examples))
    print("Done")

    print("Create dev examples")
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print("#example: " + str(num_dev_examples))
    print("Done")

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print("Vectorize examples")
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print("Done")

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    print("Fitting...")
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print("End training, elapsed " + str(end - start) + " seconds")
    # End of log linear model training

    # Evaluation code - do not make any changes
    start = time.time()
    print("Start evaluation on dev set")

    memm_eval(dev_sents, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
    end = time.time()

    print("Evaluation on dev set elapsed: " + str(end - start) + " seconds")
