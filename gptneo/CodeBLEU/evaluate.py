import argparse
import bleu
import weighted_ngram_match
import syntax_match
import dataflow_match
import json
import re
import os
from rouge import Rouge
from tree_sitter import Language, Parser


def read_json(name):
    with open(name, 'r') as f:
        json_file = json.load(f)
    return json_file


def write_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f)


def read_txt_last(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return lines[-1]


lang = 'python'
# params = '0.1,0.4,0.1,0.4'
params = '0.25,0.25,0.25,0.25'
# params = '0.1,0.1,0.4,0.4'
alpha, beta, gamma, theta = [float(x) for x in params.split(',')]


# code = 'df.columns = [\'A\' madeupword0087]'
def remove_madeupword(code):
    item_list = re.split(r'(madeupword\d{4})', code)
    item_effective = []
    for s in item_list:
        if 'madeupword' in s:
            continue
        else:
            item_effective.append(s)
    return ''.join(item_effective)


# print(item_effective)
# print(remove_madeupword(code))
# OUT: df.columns = ['A' ]

post_replace = {
    "Ġ": " ",
    "Ċ": "\n",
    "ĉ": "\t",
    'madeupword0001': '\'jupyter_string\''
}


def remove_madeupword(code):
    item_list = re.split(r'(madeupword\d{4})', code)
    item_effective = []
    for s in item_list:
        if 'madeupword' in s:
            continue
        else:
            item_effective.append(s)
    return ''.join(item_effective)


def decode(hyp):
    hyp = ''.join(hyp)
    for s, t in post_replace.items():
        hyp = hyp.replace(s, t)
    hyp = remove_madeupword(hyp)
    return hyp


def decode_string(hyp, token=True):
    if not token:
        hyp = ''.join(hyp.split(' '))
    for s, t in post_replace.items():
        hyp = hyp.replace(s, t)
    hyp = remove_madeupword(hyp)
    return hyp


# def get_devdiv_codebleu_tokenized_code(prefix):
#     hypothesis = []
#     references = []
#     reference = []
#
#     result = read_json(prefix + '/split_generation_results.json')
#     for item in result:
#         hypothesis.append(item['generation'])
#         reference.append(item['target'])
#         references.append([item['target']])
#
#     tokenized_hyps = [x.split() for x in hypothesis]
#     tokenized_refs = [[x.split()] for x in reference]
#     ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)
#     print()
#     print(tokenized_hyps[0])
#     print(tokenized_refs[0])
#
#     # calculate weighted ngram match
#     # keywords = [x.strip() for x in open('./keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]
#     keywords = [x.strip() for x in open('./CodeBLEU/keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]
#
#     def make_weights(reference_tokens, key_word_list):
#         return {token: 1 if token in key_word_list else 0.2 \
#                 for token in reference_tokens}
#
#     tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
#                                     for reference_tokens in reference] for reference in tokenized_refs]
#
#     weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)
#
#     # calculate syntax match
#     syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)
#
#     # calculate dataflow match
#     dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)
#
#     print('ngram match: {:.4f}, weighted ngram match: {:.4f}, syntax_match: {:.4f}, dataflow_match: {:.4f}'. \
#           format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))
#
#     code_bleu_score = alpha * ngram_match_score \
#                       + beta * weighted_ngram_match_score \
#                       + gamma * syntax_match_score \
#                       + theta * dataflow_match_score
#
#     print('CodeBLEU score: {:.4f}'.format(code_bleu_score))
#     return {"CodeBLEU": code_bleu_score,
#             "NgramMatch": ngram_match_score,
#             "WeightedNgramMatch": weighted_ngram_match_score,
#             "SyntaxMmatch": syntax_match_score,
#             "DataflowMatch": dataflow_match_score}
#
#
# def get_devdiv_codebleu_nontokenized_code(prefix):
#     hypothesis = []
#     references = []
#     reference = []
#
#     result = read_json(prefix + '/split_generation_results.json')
#     for item in result:
#         hypothesis.append(decode_string(item['generation']))
#         reference.append(decode_string(item['target']))
#         references.append([decode_string(item['target'])])
#
#     tokenized_hyps = [[x] for x in hypothesis]
#     tokenized_refs = [[[x]] for x in reference]
#     ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)
#     print()
#     print(tokenized_hyps[0])
#     print(tokenized_refs[0])
#
#     # calculate weighted ngram match
#     # keywords = [x.strip() for x in open('./keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]
#     keywords = [x.strip() for x in open('./CodeBLEU/keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]
#
#     def make_weights(reference_tokens, key_word_list):
#         return {token: 1 if token in key_word_list else 0.2 \
#                 for token in reference_tokens}
#
#     tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
#                                     for reference_tokens in reference] for reference in tokenized_refs]
#
#     weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)
#
#     # calculate syntax match
#     syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)
#
#     # calculate dataflow match
#     dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)
#
#     print('ngram match: {:.4f}, weighted ngram match: {:.4f}, syntax_match: {:.4f}, dataflow_match: {:.4f}'. \
#           format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))
#
#     code_bleu_score = alpha * ngram_match_score \
#                       + beta * weighted_ngram_match_score \
#                       + gamma * syntax_match_score \
#                       + theta * dataflow_match_score
#
#     print('CodeBLEU score: {:.4f}'.format(code_bleu_score))
#     return {"CodeBLEU": code_bleu_score,
#             "NgramMatch": ngram_match_score,
#             "WeightedNgramMatch": weighted_ngram_match_score,
#             "SyntaxMmatch": syntax_match_score,
#             "DataflowMatch": dataflow_match_score}


def get_devdiv_codebleu(reference_text_ori, translations_ori):
    # hypothesis = []
    # references = []
    # reference = []

    # result = read_json(prefix + '/split_generation_results.json')
    # for item in result:
    #     hypothesis.append(decode_string(item['generation']))
    #     reference.append(decode_string(item['target']))
    #     references.append([decode_string(item['target'])])

    tokenized_hyps = [x.split(' ') for x in translations_ori]
    tokenized_refs = [[x.split(' ')] for x in reference_text_ori]
    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)
    # print()
    # print(tokenized_hyps[0])
    # print(tokenized_refs[0])

    # calculate weighted ngram match
    # keywords = [x.strip() for x in open('./keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]
    keywords = [x.strip() for x in open('./CodeBLEU/keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 \
                for token in reference_tokens}

    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
                                    for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match([[i] for i in reference_text_ori], translations_ori, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match([[i] for i in reference_text_ori], translations_ori, lang)

    # print('ngram match: {:.4f}, weighted ngram match: {:.4f}, syntax_match: {:.4f}, dataflow_match: {:.4f}'. \
    #       format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

    code_bleu_score = alpha * ngram_match_score \
                      + beta * weighted_ngram_match_score \
                      + gamma * syntax_match_score \
                      + theta * dataflow_match_score

    # print('CodeBLEU score: {:.4f}'.format(code_bleu_score))
    return {"NgramM": round(ngram_match_score*100, 2),
            "WNgramM": round(weighted_ngram_match_score*100, 2),
            "SynM": round(syntax_match_score*100, 2),
            "DFM": round(dataflow_match_score*100, 2),
            "CodeBLEU": round(code_bleu_score*100, 2),}


# def get_metrics_devdiv(prefix):
#     translations_ori = []
#     reference_text_ori = []
#     all_scores = {}
#
#     print('[BLEU] Score')
#     generate_text_path = prefix + '/generate-test.txt'
#     print(read_txt_last(generate_text_path))
#     print("##############")
#
#     result = read_json(prefix + '/split_generation_results.json')
#     for item in result:
#         translations_ori.append(decode_string(item['generation']))
#         reference_text_ori.append(decode_string(item['target']))
#     em_count = 0
#     for i, j in zip(translations_ori, reference_text_ori):
#         if i == j:
#             em_count += 1
#     print('[EM] Score: {}/{} in 2000, EM: {:.4f} ({}), {:.4f} (2000)'.format(em_count, len(translations_ori),
#                                                                    em_count / len(translations_ori), len(translations_ori),
#                                                                    em_count / 2000))
#     print("##############")
#
#     hyps, refs = map(list, zip(*[[d['generation'], d['target']] for d in result]))
#     rouge = Rouge()
#     scores = rouge.get_scores(hyps, refs, avg=True)
#     print('[Rounge-1] Score [Tokenized]: Rec:{:.4f}, Pre:{:.4f}, F:{:.4f}'.format(scores['rouge-1']['r'], scores['rouge-1']['p'], scores['rouge-1']['f']))
#     # print(scores['rouge-l'])
#     hyps, refs = map(list, zip(*[[''.join(d['generation'].split(' ')), ''.join(d['target'].split(' '))] for d in result]))
#     rouge = Rouge()
#     scores = rouge.get_scores(hyps, refs, avg=True)
#     print('[Rounge-1] Score [Untokenized]: Rec:{:.4f}, Pre:{:.4f}, F:{:.4f}'.format(scores['rouge-1']['r'], scores['rouge-1']['p'], scores['rouge-1']['f']))
#     # print(scores['rouge-l'])
#     print("##############")
#
#     print('[CodeBLEU] Score [Tokenized]:')
#     get_devdiv_codebleu_tokenized_code(prefix)
#
#     print('[CodeBLEU] Score [Untokenized]:')
#     get_devdiv_codebleu_nontokenized_code(prefix)


def get_metrics_devdiv(reference_text_ori, translations_ori):
    all_scores = {}

    em_count = 0
    em_list = []
    for i, j in zip(translations_ori, reference_text_ori):
        if i == j:
            em_count += 1
            em_list.append(1)
        else:
            em_list.append(0)
    all_scores['Number'] = len(translations_ori)
    all_scores['EM(a)'] = round(em_count / 2000 * 100, 2)
    all_scores['EM(n)'] = round(em_count / len(translations_ori) * 100, 2)
    # print('[EM] Score: {}/{} in 2000, EM: {:.4f} ({}), {:.4f} (2000)'.format(em_count, len(translations_ori),
    #                                                                em_count / len(translations_ori), len(translations_ori),
    #                                                                em_count / 2000))
    # print("##############")

    # hyps, refs = map(list, zip(*[[d['generation'], d['target']] for d in result]))
    rouge = Rouge()
    scores = rouge.get_scores(translations_ori, reference_text_ori, avg=True)
    rouge_scores_list = rouge.get_scores(translations_ori, reference_text_ori)
    all_scores['R1-R'] = round(scores['rouge-1']['r']*100, 2)
    all_scores['R1-P'] = round(scores['rouge-1']['p']*100, 2)
    all_scores['R1-F'] = round(scores['rouge-1']['f']*100, 2)
    all_scores['R2-R'] = round(scores['rouge-2']['r']*100, 2)
    all_scores['R2-P'] = round(scores['rouge-2']['p']*100, 2)
    all_scores['R2-F'] = round(scores['rouge-2']['f']*100, 2)
    all_scores['RL-R'] = round(scores['rouge-l']['r']*100, 2)
    all_scores['RL-P'] = round(scores['rouge-l']['p']*100, 2)
    all_scores['RL-F'] = round(scores['rouge-l']['f']*100, 2)
    # print('[Rounge-1] Score: Rec:{:.4f}, Pre:{:.4f}, F:{:.4f}'.format(scores['rouge-1']['r'], scores['rouge-1']['p'], scores['rouge-1']['f']))
    # print("##############")

    # print('[CodeBLEU] Score:')
    all_scores.update(get_devdiv_codebleu(reference_text_ori, translations_ori))
    codebleu_scores_list = [get_devdiv_codebleu([ref], [tra]) for ref, tra in zip(reference_text_ori, translations_ori)]
    combined_list = [{"EM": i,
                      "R1-R": j['rouge-1']['r'],
                      "R1-P": j['rouge-1']['p'],
                      "R1-F": j['rouge-1']['f'],
                      "R2-R": j['rouge-2']['r'],
                      "R2-P": j['rouge-2']['p'],
                      "R2-F": j['rouge-2']['f'],
                      "RL-R": j['rouge-l']['r'],
                      "RL-P": j['rouge-l']['p'],
                      "RL-F": j['rouge-l']['f'],
                      "NgramM": k['NgramM'],
                      "WNgramM": k['WNgramM'],
                      "SynM": k['SynM'],
                      "DFM": k['DFM'],
                      "CodeBLEU": k['CodeBLEU'],
                      "reference_text": x,
                      "translation_text": y,
                      }for i, j, k, x, y in zip(em_list, rouge_scores_list, codebleu_scores_list, reference_text_ori, translations_ori)]
    return all_scores, combined_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # '/home/t-jhuang/home/save/test/new_checkpoint/result/split_generation_results.json'
    parser.add_argument("--generation_dir",
                        default='/home/t-jhuang/home/save/test/new_checkpoint/result',
                        type=str, required=False, help="Path to save the splited generation results.")
    parser.add_argument("--result_length", action='store_true', help="Whether to split data by length and eval.")
    args = parser.parse_args()

    translations_ori, reference_text_ori = [], []
    translations_tok, reference_text_tok = [], []
    result = read_json(args.generation_dir + '/split_generation_results.json')
    for item in result:
        if len(''.join(decode_string(item['generation'], token=False).split(' '))) == 0 or len(''.join(decode_string(item['generation'], token=True).split(' '))) == 0:
            continue
        translations_ori.append(decode_string(item['generation'], token=False))
        reference_text_ori.append(decode_string(item['target'], token=False))
        translations_tok.append(decode_string(item['generation'], token=True))
        reference_text_tok.append(decode_string(item['target'], token=True))

    print("$$$$$$$$$$$$$$$$$$")
    print("{},{},{},{},".format(len([item for item in translations_ori if len(''.join(item.split(' ')))==0]),
                                len([item for item in reference_text_ori if len(''.join(item.split(' ')))==0]),
                                len([item for item in translations_tok if len(''.join(item.split(' ')))==0]),
                                len([item for item in reference_text_tok if len(''.join(item.split(' ')))==0]), ))
    print("$$$$$$$$$$$$$$$$$$")
    print("All data ori: {}".format(len(reference_text_ori)))
    results, combined_list = get_metrics_devdiv(reference_text_ori, translations_ori)
    write_json([results, combined_list], os.path.join(args.generation_dir, 'surface_form_score_ori.json'))
    print("Results All dev/test original string")
    print("\t".join(results.keys()))
    print("\t".join([str(i) for i in results.values()]))

    print("$$$$$$$$$$$$$$$$$$")
    print("All data tokenized: {}".format(len(reference_text_tok)))
    results, combined_list = get_metrics_devdiv(reference_text_tok, translations_tok)
    write_json([results, combined_list], os.path.join(args.generation_dir, 'surface_form_score_token.json'))
    print("Results All dev/test tokenized version")
    print("\t".join(results.keys()))
    print("\t".join([str(i) for i in results.values()]))

    if args.result_length:
        reference_easy, reference_medium, reference_hard = [], [], []
        translation_easy, translation_medium, translation_hard = [], [], []
        for r_i, t_i in zip(reference_text_ori, translations_ori):
            num_line = len(r_i.split('\n'))
            if num_line == 1:
                reference_easy.append(r_i)
                translation_easy.append(t_i)
            elif num_line <= 4:
                reference_medium.append(r_i)
                translation_medium.append(t_i)
            else:
                reference_hard.append(r_i)
                translation_hard.append(t_i)

        print("$$$$$$$$$$$$$$$$$$")
        print("EASY data: {}".format(len(reference_easy)))
        results, combined_list = get_metrics_devdiv(reference_easy, translation_easy)
        print("Results EASY dev/test original string")
        print("\t".join(results.keys()))
        print("\t".join([str(i) for i in results.values()]))
        print("$$$$$$$$$$$$$$$$$$")
        print("MEDIUM data: {}".format(len(reference_medium)))
        results, combined_list = get_metrics_devdiv(reference_medium, translation_medium)
        print("Results MEDIUM dev/test original string")
        print("\t".join(results.keys()))
        print("\t".join([str(i) for i in results.values()]))
        print("$$$$$$$$$$$$$$$$$$")
        print("HARD data: {}".format(len(reference_hard)))
        results, combined_list = get_metrics_devdiv(reference_hard, translation_hard)
        print("Results HARD dev/test original string")
        print("\t".join(results.keys()))
        print("\t".join([str(i) for i in results.values()]))


