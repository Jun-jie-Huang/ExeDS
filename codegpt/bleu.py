# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def _bleu(ref_file, trans_file, subword_option=None):
    max_order = 4
    smooth = True
    # ref_files = [ref_file]
    # reference_text = []
    with open(ref_file, 'r', encoding='utf-8') as fh:
        reference_text = fh.read().split('##########\n')
        print("reference_text: {}".format(len(reference_text)))
    with open(trans_file, 'r', encoding='utf-8') as fh:
        translation_text = fh.read().split('##########\n')
        print("translation_text: {}".format(len(translation_text)))
    if reference_text[-1] == '' and translation_text[-1] == '':
        reference_text = reference_text[:-1]
        translation_text = translation_text[:-1]
        print("reference_text new: {}".format(len(reference_text)))
        print("translation_text new: {}".format(len(translation_text)))
    reference_list = [[[item]] for item in reference_text]
    translation_list = [[item] for item in translation_text]

    bleu_score, _, _, _, _, _ = compute_bleu(reference_list, translation_list, max_order, smooth)
    # for reference_filename in ref_files:
    #     with open(reference_filename) as fh:
    #         reference_text.append(fh.readlines())
    # per_segment_references = []
    # for references in zip(*reference_text):
    #     reference_list = []
    #     for reference in references:
    #         reference_list.append(reference.strip().split())
    #     per_segment_references.append(reference_list)
    # translations = []
    # with open(trans_file) as fh:
    #     for line in fh:
    #         translations.append(line.strip().split())
    # bleu_score, _, _, _, _, _ = compute_bleu(per_segment_references, translations, max_order, smooth)
    return round(100 * bleu_score, 2)


# import CodeBLEU.bleu as bleu
# import CodeBLEU.weighted_ngram_match as weighted_ngram_match
# import CodeBLEU.syntax_match as syntax_match
# import CodeBLEU.dataflow_match as dataflow_match
# from rouge import Rouge
# from tree_sitter import Language, Parser
#
#
# def _evaluation_untokenize(ref_file, trans_file, subword_option=None):
#     max_order = 4
#     lang = 'python'
#     smooth = True
#     params = '0.25,0.25,0.25,0.25'
#     # params = '0.1,0.1,0.4,0.4'
#     alpha, beta, gamma, theta = [float(x) for x in params.split(',')]
#
#     with open(ref_file, 'r') as fh:
#         reference_text = fh.read().split('##########\n')
#     print("reference_text: {}".format(len(reference_text)))
#     reference_list = [[[item]] for item in reference_text]
#     with open(trans_file) as fh:
#         translation_text = fh.read().split('##########\n')
#     print("translation_text: {}".format(len(translation_text)))
#     translation_list = [[item] for item in translation_text]
#     bleu_score, _, _, _, _, _ = compute_bleu(reference_list, translation_list, max_order, smooth)
#
#     # codebleu
#     ngram_match_score = bleu.corpus_bleu(reference_list, translation_list)
#     keywords = [x.strip() for x in open('./CodeBLEU/keywords/python.txt', 'r', encoding='utf-8').readlines()]
#     def make_weights(reference_tokens, key_word_list):
#         return {token: 1 if token in key_word_list else 0.2 \
#                 for token in reference_tokens}
#     tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
#                                     for reference_tokens in reference] for reference in reference_list]
#     weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, translation_list)
#     syntax_match_score = syntax_match.corpus_syntax_match(reference_text, translation_text, lang)
#     dataflow_match_score = dataflow_match.corpus_dataflow_match(reference_text, translation_text, lang)
#
#     print('ngram match: {:.4f}, weighted ngram match: {:.4f}, syntax_match: {:.4f}, dataflow_match: {:.4f}'. \
#           format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))
#     code_bleu_score = alpha * ngram_match_score \
#                       + beta * weighted_ngram_match_score \
#                       + gamma * syntax_match_score \
#                       + theta * dataflow_match_score
#     print('CodeBLEU score: {:.4f}'.format(code_bleu_score))
#     return {"BLEU": bleu_score,
#             "CodeBLEU": code_bleu_score,
#             "NgramMatch": ngram_match_score,
#             "WeightedNgramMatch": weighted_ngram_match_score,
#             "SyntaxMmatch": syntax_match_score,
#             "DataflowMatch": dataflow_match_score}

