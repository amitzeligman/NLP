import re
from numpy import array
from collections import defaultdict, Counter

def regex_rule(regex):
    pattern = re.compile(regex)
    return lambda s: bool(pattern.match(s))

# Note - order of rules is important, first rule to match is taken
rare_words_transformation_rules = [
    ('twoDigitNum',        regex_rule('^\d{2}$')),                 # Example: 12
    ('fourDigitNum',       regex_rule('^\d{4}$')),                 # Example: 1234
    ('otherNum',           regex_rule('^\d+\.?\d+$')),             # Examples: 12345, 1234.235, 1.3
    ('commaInNum',         regex_rule('^(?:\d+,)+\d+\.?\d+$')),    # Examples: '1,234', '1,234.14'
    ('dashAndNum',         regex_rule('^[0-9]+[0-9\-]+[0-9]+$')),  # Examples: 1-1, 07-05-1998
    ('slashAndNum',        regex_rule('^[0-9]+[0-9/]+[0-9]+$')),   # Example: 10/10/2010, 1000/1000/201100
    ('ordinal',            regex_rule('^\d+(nd|rd|th)$')),         # Examples: 22nd, 53rd, 14th
    ('hour:minute',        regex_rule('^\d{1,2}:\d{2}')),          # Examples: 3:15, 12:49

    ('allCaps',            regex_rule('^[A-Z]+$')),                # Example: ALLCAP
    ('capPeriod',          regex_rule('^[A-Z]\.$')),               # Example: M.

    ('thing-and-thing',    regex_rule('^[a-z]+\-and\-[a-z]+$')),   # Example: black-and-white
    ('thing-and-thing',    regex_rule('^[a-z]+\-and\-[a-z]+$')),   # Example: black-and-white
    ('thing-than-thing',   regex_rule('^[a-z]+\-than\-[a-z]+$')),  # Example: smaller-than-expected
    ('thing-the-thing',    regex_rule('^[a-z]+\-the\-[a-z]+$')),   # Example: behind-the-scenes
    ('co-thing',               regex_rule('^co\-[a-z]+$')),        # Example: co-sponsored
    ('pre-thing',              regex_rule('^pre\-[a-z]+$')),       # Example: pre-empt
    ('pro-thing',              regex_rule('^pro\-[a-z]+$')),       # Example: pro-active
    ('much-thing',             regex_rule('^much\-[a-z]+$')),      # Example: much-publicized
    ('most-thing',             regex_rule('^most\-[a-z]+$')),      # Example: most-active
    ('low-thing',              regex_rule('^low\-[a-z]+$')),       # Example: low-level
    ('high-thing',             regex_rule('^high\-[a-z]+$')),      # Example: high-visibility
    ('inter-thing',            regex_rule('^inter\-[a-z]+$')),     # Example: inter-city
    ('-a-',                    regex_rule('^.+\-a\-.+$')),         # Example: 18-a-share

    ('iedLowercase',           regex_rule('^[a-z]+ied$')),         # Example: supplied
    ('edLowercase',            regex_rule('^[a-z]+ed$')),          # Example: played
    ('ingLowercase',           regex_rule('^[a-z]+ing$')),         # Example: playing
    ('tionLowercase',          regex_rule('^[a-z]+tion$')),        # Example: transition
    ('sionLowercase',          regex_rule('^[a-z]+sion$')),        # Example: emission
    ('xionLowercase',          regex_rule('^[a-z]+xion$')),        # Example: complexion
    ('ableLowercase',          regex_rule('^[a-z]+able$')),        # Example: formidable
    ('ibleLowercase',          regex_rule('^[a-z]+ible$')),        # Example: tangible
    ('fulLowercase',           regex_rule('^[a-z]+ful$')),         # Example: powerful
    ('anceLowercase',          regex_rule('^[a-z]+ance$')),        # Example: performance
    ('enceLowercase',          regex_rule('^[a-z]+ence$')),        # Example: intelligence
    ('sialLowercase',          regex_rule('^[a-z]+sial$')),        # Example: controversial
    ('tialLowercase',          regex_rule('^[a-z]+tial$')),        # Example: potential
    ('mentLowercase',          regex_rule('^[a-z]+ment$')),        # Example: establishment
    ('shipLowercase',          regex_rule('^[a-z]+ship$')),        # Example: relationship
    ('nessLowercase',          regex_rule('^[a-z]+ness$')),        # Example: kindness
    ('hoodLowercase',          regex_rule('^[a-z]+hood$')),        # Example: neighborhood
    ('domLowercase',           regex_rule('^[a-z]+dom$')),         # Example: kingdom
    ('eeLowercase',            regex_rule('^[a-z]+ee$')),          # Example: trainee
    ('istLowercase',           regex_rule('^[a-z]+ist$')),         # Example: socialist
    ('ismLowercase',           regex_rule('^[a-z]+ism$')),         # Example: capitalism
    ('ageLowercase',           regex_rule('^[a-z]+age$')),         # Example: village
    ('erLowercase',            regex_rule('^[a-z]+er$')),          # Example: driver
    ('orLowercase',            regex_rule('^[a-z]+or$')),          # Example: director
    ('ityLowercase',           regex_rule('^[a-z]+ity$')),         # Example: equality
    ('tyLowercase',            regex_rule('^[a-z]+ty$')),          # Example: cruelty
    ('ryLowercase',            regex_rule('^[a-z]+ry$')),          # Example: robbery
    ('lyLowercase',            regex_rule('^[a-z]+ly$')),          # Example: easily
    ('wardLowercase',          regex_rule('^[a-z]+ward$')),        # Example: backward
    ('wardsLowercase',         regex_rule('^[a-z]+wards$')),       # Example: backwards
    ('izeLowercase',           regex_rule('^[a-z]+ize$')),         # Example: characterize
    ('iseLowercase',           regex_rule('^[a-z]+ise$')),         # Example: characterise (UK)
    ('ifyLowercase',           regex_rule('^[a-z]+ify$')),         # Example: signify
    ('ateLowercase',           regex_rule('^[a-z]+ate$')),         # Example: irrigate
    ('enLowercase',            regex_rule('^[a-z]+en$')),          # Example: strengthen
    ('icLowercase',            regex_rule('^[a-z]+ic$')),          # Example: classic
    ('alLowercase',            regex_rule('^[a-z]+al$')),          # Example: brutal
    ('yLowercase',             regex_rule('^[a-z]+y$')),           # Example: cloudy
    ('estLowercase',           regex_rule('^[a-z]+est$')),         # Example: strongest
    ('ianLowercase',           regex_rule('^[a-z]+ian$')),         # Example: utilitarian
    ('iveLowercase',           regex_rule('^[a-z]+ive$')),         # Example: productive
    ('ishLowercase',           regex_rule('^[a-z]+ish$')),         # Example: childish
    ('lessLowercase',          regex_rule('^[a-z]+less$')),        # Example: useless
    ('ousLowercase',           regex_rule('^[a-z]+ous$')),         # Example: nervous

    ('otherLowercase',         regex_rule('^[a-z]+$')),            # Example: abc

    ('eseNationality',         regex_rule('^[A-Z][a-z]+ese$')),    # Example: Japanese
    ('ishNationality',         regex_rule('^[A-Z][a-z]+ish$')),    # Example: Spanish
    ('ianNationality',         regex_rule('^[A-Z][a-z]+ian$')),    # Example: Canadian

    ('initCap_iedLowercase',   regex_rule('^[A-Z][a-z]+ied$')),    # Example: Supplied
    ('initCap_edLowercase',    regex_rule('^[A-Z][a-z]+ed$')),     # Example: Played
    ('initCap_ingLowercase',   regex_rule('^[A-Z][a-z]+ing$')),    # Example: Playing
    ('initCap_tionLowercase',  regex_rule('^[A-Z][a-z]+tion$')),   # Example: Transition
    ('initCap_sionLowercase',  regex_rule('^[A-Z][a-z]+sion$')),   # Example: Emission
    ('initCap_xionLowercase',  regex_rule('^[A-Z][a-z]+xion$')),   # Example: Complexion
    ('initCap_ableLowercase',  regex_rule('^[A-Z][a-z]+able$')),   # Example: Formidable
    ('initCap_ibleLowercase',  regex_rule('^[A-Z][a-z]+ible$')),   # Example: Tangible
    ('initCap_fulLowercase',   regex_rule('^[A-Z][a-z]+ful$')),    # Example: Powerful
    ('initCap_anceLowercase',  regex_rule('^[A-Z][a-z]+ance$')),   # Example: Performance
    ('initCap_enceLowercase',  regex_rule('^[A-Z][a-z]+ence$')),   # Example: Intelligence
    ('initCap_sialLowercase',  regex_rule('^[A-Z][a-z]+sial$')),   # Example: Controversial
    ('initCap_tialLowercase',  regex_rule('^[A-Z][a-z]+tial$')),   # Example: Potential
    ('initCap_mentLowercase',  regex_rule('^[A-Z][a-z]+ment$')),   # Example: Establishment
    ('initCap_shipLowercase',  regex_rule('^[A-Z][a-z]+ship$')),   # Example: Relationship
    ('initCap_nessLowercase',  regex_rule('^[A-Z][a-z]+ness$')),   # Example: Kindness
    ('initCap_hoodLowercase',  regex_rule('^[A-Z][a-z]+hood$')),   # Example: Neighborhood
    ('initCap_domLowercase',   regex_rule('^[A-Z][a-z]+dom$')),    # Example: Kingdom
    ('initCap_eeLowercase',    regex_rule('^[A-Z][a-z]+ee$')),     # Example: Trainee
    ('initCap_istLowercase',   regex_rule('^[A-Z][a-z]+ist$')),    # Example: Socialist
    ('initCap_ismLowercase',   regex_rule('^[A-Z][a-z]+ism$')),    # Example: Capitalism
    ('initCap_ageLowercase',   regex_rule('^[A-Z][a-z]+age$')),    # Example: Village
    ('initCap_erLowercase',    regex_rule('^[A-Z][a-z]+er$')),     # Example: Driver
    ('initCap_orLowercase',    regex_rule('^[A-Z][a-z]+or$')),     # Example: Director
    ('initCap_ityLowercase',   regex_rule('^[A-Z][a-z]+ity$')),    # Example: Equality
    ('initCap_tyLowercase',    regex_rule('^[A-Z][a-z]+ty$')),     # Example: Cruelty
    ('initCap_ryLowercase',    regex_rule('^[A-Z][a-z]+ry$')),     # Example: Robbery
    ('initCap_lyLowercase',    regex_rule('^[A-Z][a-z]+ly$')),     # Example: Easily
    ('initCap_wardLowercase',  regex_rule('^[A-Z][a-z]+ward$')),   # Example: Backward
    ('initCap_wardsLowercase', regex_rule('^[A-Z][a-z]+wards$')),  # Example: Backwards
    ('initCap_izeLowercase',   regex_rule('^[A-Z][a-z]+ize$')),    # Example: Characterize
    ('initCap_iseLowercase',   regex_rule('^[A-Z][a-z]+ise$')),    # Example: Characterise (UK)
    ('initCap_ifyLowercase',   regex_rule('^[A-Z][a-z]+ify$')),    # Example: Signify
    ('initCap_ateLowercase',   regex_rule('^[A-Z][a-z]+ate$')),    # Example: Irrigate
    ('initCap_enLowercase',    regex_rule('^[A-Z][a-z]+en$')),     # Example: Strengthen
    ('initCap_icLowercase',    regex_rule('^[A-Z][a-z]+ic$')),     # Example: Classic
    ('initCap_alLowercase',    regex_rule('^[A-Z][a-z]+al$')),     # Example: Brutal
    ('initCap_yLowercase',     regex_rule('^[A-Z][a-z]+y$')),      # Example: Cloudy

    ('initCap', regex_rule('^[A-Z].*$'))  # Example: Cap
]

MIN_FREQ = 3
def invert_dict(d):
    res = {}
    for k, v in d.items():
        res[v] = k
    return res

def read_conll_pos_file(path):
    """
        Takes a path to a file and returns a list of word/tag pairs
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                tokens = line.strip().split("\t")
                curr.append((tokens[1],tokens[3]))
    return sents

def read_conll_ner_file(path):
    """
    Reads a path to a file @path in CoNLL file format.
    @returns a list of examples [(tokens), (labels)]. @tokens and @labels are lists of string.
    """
    sents = []
    with open(path, "r") as fstream:
        curr = []
        for line in fstream:
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(curr) > 0:
                    sents.append(curr)
                    curr = []
            else:
                assert "\t" in line, r"Invalid CONLL format; expected a '\t' in {}".format(line)
                tok, lbl = line.split("\t")
                curr.append((tok, lbl))
    return sents

NER_LBLS = ["PER", "ORG", "LOC", "MISC", "O"]
def evaluate_ner(gold_tag_seqs, pred_tag_seqs):
    """Evaluates model performance on @examples.

    This function uses the model to predict labels for @examples and constructs a confusion matrix.

    Returns:
        The F1 score for predicting tokens as named entities.
    """
    token_cm = ConfusionMatrix(labels=NER_LBLS)

    correct_preds, total_correct, total_preds = 0., 0., 0.
    for gold_tags, pred_tags  in zip(gold_tag_seqs, pred_tag_seqs):
        for l, l_ in zip(gold_tags, pred_tags):
            token_cm.update(NER_LBLS.index(l), NER_LBLS.index(l_))
        gold = set(get_chunks(gold_tags))
        pred = set(get_chunks(pred_tags))
        correct_preds += len(gold.intersection(pred))
        total_preds += len(pred)
        total_correct += len(gold)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    print("Token-level confusion matrix:\n" + token_cm.as_table())
    print("Token-level scores:\n" + token_cm.summary())
    print("Entity level P/R/F1: {:.2f}/{:.2f}/{:.2f}".format(*(p, r, f1)))
    return token_cm, (p, r, f1)

def to_table(data, row_labels, column_labels, precision=2, digits=4):
    """Pretty print tables.
    Assumes @data is a 2D array and uses @row_labels and @column_labels
    to display table.
    """
    # Convert data to strings
    data = [["%04.2f"%v for v in row] for row in data]
    cell_width = max(
        max(map(len, row_labels)),
        max(map(len, column_labels)),
        max(max(map(len, row)) for row in data))
    def c(s):
        """adjust cell output"""
        return s + " " * (cell_width - len(s))
    ret = ""
    ret += "\t".join(map(c, column_labels)) + "\n"
    for l, row in zip(row_labels, data):
        ret += "\t".join(map(c, [l] + row)) + "\n"
    return ret

class ConfusionMatrix(object):
    """
    A confusion matrix stores counts of (true, guessed) labels, used to
    compute several evaluation metrics like accuracy, precision, recall
    and F1.
    """

    def __init__(self, labels, default_label=None):
        self.labels = labels
        self.default_label = default_label if default_label is not None else len(labels) -1
        self.counts = defaultdict(Counter)

    def update(self, gold, guess):
        """Update counts"""
        self.counts[gold][guess] += 1

    def as_table(self):
        """Print tables"""
        # Header
        data = [[self.counts[l][l_] for l_,_ in enumerate(self.labels)] for l,_ in enumerate(self.labels)]
        return to_table(data, self.labels, ["go\\gu"] + self.labels)

    def summary(self, quiet=False):
        """Summarize counts"""
        keys = range(len(self.labels))
        data = []
        macro = array([0., 0., 0., 0.])
        micro = array([0., 0., 0., 0.])
        default = array([0., 0., 0., 0.])
        for l in keys:
            tp = self.counts[l][l]
            fp = sum(self.counts[l_][l] for l_ in keys if l_ != l)
            tn = sum(self.counts[l_][l__] for l_ in keys if l_ != l for l__ in keys if l__ != l)
            fn = sum(self.counts[l][l_] for l_ in keys if l_ != l)

            acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
            prec = (tp)/(tp + fp) if tp > 0  else 0
            rec = (tp)/(tp + fn) if tp > 0  else 0
            f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0

            # update micro/macro averages
            micro += array([tp, fp, tn, fn])
            macro += array([acc, prec, rec, f1])
            if l != self.default_label: # Count count for everything that is not the default label!
                default += array([tp, fp, tn, fn])

            data.append([acc, prec, rec, f1])

        # micro average
        tp, fp, tn, fn = micro
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])
        # Macro average
        data.append(macro / len(keys))

        # default average
        tp, fp, tn, fn = default
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])

        # Macro and micro average.
        return to_table(data, self.labels + ["micro","macro","not-O"], ["label", "acc", "prec", "rec", "f1"])

def get_chunks(seq, default=NER_LBLS.index("O")):
    """Breaks input of 4 4 4 0 0 4 0 ->   (0, 4, 5), (0, 6, 7)"""
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None
        # End of a chunk + start of a chunk!
        elif tok != default:
            if chunk_type is None:
                chunk_type, chunk_start = tok, i
            elif tok != chunk_type:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks

def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1

def compute_vocab_count(sents):
    """
        Takes a corpus and computes all words and the number of times they appear
    """
    vocab = {}
    for sent in sents:
        for token in sent:
            increment_count(vocab, token[0])
    return vocab

def replace_word(word):
    """
        Replaces rare words with categories (numbers, dates, etc...)
    """
    ### YOUR CODE HERE

    def all_numbers(word_):
        word_ = list(word_)
        if all(map(lambda p: p.isdigit(), word_)):
            return True
        else:
            return False

    # All numbers cases
    if all_numbers(word):
        if len(word) == 2:
            return '2DigitNum'
        elif len(word) == 4:
            return '4DigitNum'
        else:
            return 'num'
    if all_numbers(word.replace(',', '')):
        return 'numWithComma'

    elif '-' in word:
        if all_numbers(word.replace('_', '')) or all_numbers(word.replace('/', '')):
            return 'date'
    elif '.' in word:
        if all_numbers(word.replace('.', '')):
            return 'containsDigitAndPeriod'
    elif '.' in word and ',' in word:
        if all_numbers(word.replace(',', '').replace('.', '')):
            return 'containsDigitAndComma'

    if not any(map(lambda p: p.isdigit(), list(word))):

        if '-' in word:
            return 'dualWord'

        # All capital
        if word.upper() == word:
            return 'allCaps'

        # Non capital
        elif word.lower() == word:
            return 'unCap'
        # Init cap
        elif word.capitalize() == word:
            return 'initCap'

    ### YOUR CODE HERE
    return "UNK"

def preprocess_sent(vocab, sents):
    """
        return a sentence, where every word that is not frequent enough is replaced
    """
    res = []
    total, replaced = 0, 0
    for sent in sents:
        new_sent = []
        for token in sent:
            if token[0] in vocab and vocab[token[0]] >= MIN_FREQ:
                new_sent.append(token)
            else:
                new_sent.append((replace_word(token[0]), token[1]))
                replaced += 1
            total += 1
        res.append(new_sent)
    print("replaced: " + str(float(replaced)/total))
    return res
