import os
import re
import argparse
import json
import csv
import copy
import platform
import ast
import time
from collections import Counter


def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=1)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

# csv read
def csv_reader(path):
    with open(path, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp)
        data = [i for i in reader]
    return data
def csv_writer(path, header, data):
    with open(path, 'w', encoding='utf-8', newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)


def compare_code_string(code1, code2):
    code1, code2 = ''.join(code1.split(' ')), ''.join(code2.split(' '))
    code1, code2 = code1.replace('\"', '\''), code2.replace('\"', '\'')
    return code1 == code2


def replace_base_path(cells, old_path, new_path):
    new_cells = []
    for cell in cells:
        new_cell = copy.deepcopy(cell)
        string = '[SPLIT]'.join(cell['source'])
        string = re.sub("\[DATA_ROOT\]", new_path, string)
        string = string.replace(old_path, new_path)
        new_cell['source'] = string.split('[SPLIT]')
        new_cells.append(new_cell)
    return new_cells


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


def decode_string(hyp):
    hyp=''.join(hyp.split(' '))
    if len(hyp) == 0:
        return ''
    if hyp[-1] == '\n':
        hyp = hyp[:-1]
    for s, t in post_replace.items():
        hyp = hyp.replace(s, t)
    hyp = remove_madeupword(hyp)
    hyp = re.sub(' +', ' ', hyp)
    return hyp

# def decode_string(hyp):
#     if len(hyp) == 0:
#         return ''
#     if hyp[-1] == '\n':
#         hyp = hyp[:-1]
#     for s, t in post_replace.items():
#         hyp = hyp.replace(s, t)
#     hyp = remove_madeupword(hyp)
#     return hyp


indent_template1 = re.compile(r' {4}(.+)')
import_template1 = re.compile(r'from(.+)import(.+)as(.+)$')
import_template2 = re.compile(r'import(.+)as(.+)$')
import_template3 = re.compile(r'from(.+)import(.+)$')
import_template4 = re.compile(r'import(.+)$')
for_template1 = re.compile(r'(.*)for(.+)in(.+):(.*)$')
for_template2 = re.compile(r'(.*)[\[\{](.*)for(.+)in(.+)[\]\}](.*)$')
lambda_template1 = re.compile(r'(.+)lambda(.+):(.+)$')
def merge_generation_code(generation):
    # return ''.join(generation.split(' '))
    codes = generation.split('\n')
    new_codes = []
    if len(codes) > 0:
        codes[:-1] = [item + '\n' for item in codes[:-1]]
        for code in codes:
            temp = ''.join(code.split(' '))
            if indent_template1.match(code):
                temp = '    ' + temp
            # print("temp 1: {}".format(temp))
            if import_template1.match(temp):
                match = import_template1.match(temp)
                temp = 'from '+match.group(1)+' import '+match.group(2)+' as '+match.group(3)
            elif import_template2.match(temp):
                match = import_template2.match(temp)
                temp = 'import '+match.group(1)+' as '+match.group(2)
            elif import_template3.match(temp):
                match = import_template3.match(temp)
                temp = 'from '+match.group(1)+' import '+match.group(2)
            elif import_template4.match(temp):
                match = import_template4.match(temp)
                temp = 'import '+match.group(1)
            elif for_template1.match(temp):
                match = for_template1.match(temp)
                _indi = 0 if len(match.group(1)) == 0 else 1
                temp = match.group(1)+' '*_indi+'for '+match.group(2)+' in '+match.group(3)+':'+match.group(4)
            elif for_template2.match(temp):
                match = for_template2.match(temp)
                temp = match.group(1)+temp[len(match.group(1))]+match.group(2)+' for '+match.group(3)+' in '+\
                       match.group(4)+temp[6+len(match.group(1))+len(match.group(2))+len(match.group(3))+len(match.group(4))]+match.group(5)
            elif lambda_template1.match(temp):
                match = lambda_template1.match(temp)
                temp = match.group(1)+' lambda '+match.group(2)+':'+match.group(3)
            # print("temp 2: {}".format(temp))
            new_codes.append(temp)
        # return ''.join(new_codes)
    return '\n'.join(new_codes)


def replace_file_dir_path(cells, old_path, new_path):
    new_cells = []
    for cell in cells:
        new_cell = copy.deepcopy(cell)
        string = '[SPLIT]'.join(cell['source'])
        string = string.replace(old_path, new_path)
        new_cell['source'] = string.split('[SPLIT]')
        new_cells.append(new_cell)
    return new_cells


def obtain_cell_output(notebook, index):
    if notebook['cells'][1]['source'][0] == '%load_ext dumpvar_0_extension':
        row_id = int(index['row_id'])+2
    else:
        row_id = int(index['row_id'])
    cell = notebook['cells'][row_id]
    output = {}
    output['idx'] = index['idx']
    output['dir'] = index['dir']
    output['nbid'] = index['nbid']
    output['exception'] = cell['metadata']['papermill']['exception']
    if not output['exception']:
        if len(cell['outputs']) > 0:
            output_types = [out['output_type'] for out in cell['outputs']]
            # print("{}, {}".format(len(output_types), output_types))
            if 'stream' in output_types:
                output['output_type'] = 'stream'
                output['text'] = cell['outputs'][output_types.index('stream')].get('text', [])
            elif 'execute_result' in output_types:
                output['output_type'] = 'execute_result'
                if 'text/plain' in cell['outputs'][output_types.index('execute_result')]['data']:
                    output['text'] = cell['outputs'][output_types.index('execute_result')]['data'].get('text/plain', [])
                elif 'data' in cell['outputs'][output_types.index('execute_result')]:
                    output['text'] = cell['outputs'][output_types.index('execute_result')]['data'].get('text/plain', [])
            elif 'error' in output_types:
                output['output_type'] = 'error'
                _index = 0
                if len(cell['outputs']) > 0:
                    for _i, _out in enumerate(cell['outputs']):
                        if 'ename' in _out and 'evalue' in _out:
                            _index = _i
                            break
                try:
                    output['text'] = cell['outputs'][_index]['ename'] + ': ' + cell['outputs'][_index]['evalue']
                except:
                    output['text'] = ''
            else:
                output['output_type'] = ''.join(output_types)
                output['text'] = ''
        else:
            output['output_type'] = 'None'
            output['text'] = ''
    else:
        output['output_type'] = 'error'
        _index = 0
        if len(cell['outputs']) > 0:
            for _i, _out in enumerate(cell['outputs']):
                if 'ename' in _out and 'evalue' in _out:
                    _index = _i
                    break
        try:
            output['text'] = cell['outputs'][_index]['ename'] + ': ' + cell['outputs'][_index]['evalue']
        except:
            output['text'] = cell['outputs'][_index]
    output['code'] = cell['source']
    return output


def compute_f1(pred_toks, gold_toks):
    try:
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
    except:
        common = [item_pred for item_pred in pred_toks if item_pred in gold_toks]
        num_same = len(common)
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_macro_f1(preds, gold_list):
    # preds: a list of list
    # gold_list: a list
    return max([compute_f1(pred_list, gold_list) for pred_list in preds])


def strict_correct_match(prediction_text, result):
    # return true if correct, else False
    correct = False
    answers, predictions = [], []
    _truth = [prediction_text==result['ans1'], prediction_text==result['ans2'], prediction_text==result['ans3']]
    if any(_truth):
        answers = [result['ans{}'.format(idx)] for idx in range(1, 4) if len(result['ans{}'.format(idx)])>0]
        predictions = [[prediction_text]]
        correct = True
    elif prediction_text[-1] == '\n':
        _line_truth = [prediction_text[:-1]==result['ans1'], prediction_text[:-1]==result['ans2'], prediction_text[:-1]==result['ans3']]
        if any(_line_truth):
            answers = [result['ans{}'.format(idx)] for idx in range(1, 4) if len(result['ans{}'.format(idx)])>0]
            predictions = [[prediction_text[:-1]]]
            correct = True
    return {'strict_correct': correct,
            'correct': correct,
            'partial_correct': correct,
            'f1': 1.0 if correct else 0.0,
            'answers': answers,
            'predictions': predictions}


def re_extract_digits(string):
    numbers = re.findall(r"\-?\d+\.?\d*", string)
    real_numbers = []
    for num in numbers:
        real_numbers.append(float(num))
    return real_numbers


def obtain_answer_number(result):
    number_answer = []
    if result['ans1']:
        number_answer.append([round(i, 2) for i in re_extract_digits(result['ans1'])])
    if result['ans2']:
        number_answer.append([round(i, 2) for i in re_extract_digits(result['ans2'])])
    if result['ans3']:
        number_answer.append([round(i, 2) for i in re_extract_digits(result['ans3'])])
    # number_answer = list(set([j for i in number_answer for j in i]))
    return number_answer


def obtain_prediction_number(prediction_text):
    number_predictions = []
    number_prediction = re_extract_digits(prediction_text)
    number_predictions.append([round(i, 2) for i in number_prediction])
    if len(number_prediction) == 1 and 0<number_prediction[0]<1:
        number_predictions.append([round(number_prediction[0]*100, 2)])
    if len(number_prediction) == 1 and 1<number_prediction[0]<100:
        number_predictions.append([round(number_prediction[0]/100, 2)])
    return number_predictions


def re_extract_tuple(string):
    if len(string) == 0:
        return []
    try:
        lines = string.split('\n')
        tup = [re_extract_digits(l) for l in lines if len(re_extract_digits(l))>0]
        # tup = [[round(j, 2) for j in i] for i in tup]
    except:
        tup = []
    return tup


def re_extract_array_string(string):
    if len(string) == 0:
        return []
    try:
        list_string = ast.literal_eval(string)  # TODO debug
        if type(list_string) == tuple:
            list_string = list(list_string)
        else:
            list_string = [list_string]
    except:
        # list_string = re.split(r'[\'\n\",;\(\)\[\]\s]', string)  # add space
        list_string = re.split(r'[\'\n\",;\(\)\[\]]', string)
        list_string = [i for i in list_string if len(''.join(i.split(' '))) > 0 ]
        # list_string = []
    return list_string


def obtain_answer_array(result):
    if result['answer_type'] == 'array number':
        answers = obtain_answer_number(result)
    elif result['answer_type'] == 'array tuple':
        answers = [[round(j, 2) for i in re_extract_tuple(result['ans{}'.format(idx)]) for j in i] for idx in range(1, 4) if result['ans{}'.format(idx)]]
    elif result['answer_type'] == 'array str':
        answers = [re_extract_array_string(result['ans{}'.format(i)]) for i in range(1, 4) if result['ans{}'.format(i)]]
    else:
        answers = [[]]
    return answers


def obtain_prediction_array(prediction_text, answer_type):
    if answer_type == 'array number' or answer_type == 'array tuple':
        predictions = []
        number_prediction = re_extract_digits(prediction_text)
        predictions.append([round(i, 2) for i in number_prediction])
    elif answer_type == 'array str':
        predictions = re_extract_array_string(prediction_text)
        if not all([type(i)==list for i in predictions]):
            try:
                predictions2 = [j for item in predictions for j in re.split(r'[ ]', item) if len(j)>0]
                predictions2 = [i if type(i)==list else [i] for i in predictions2] + [predictions2]
                predictions = [i if type(i)==list else [i] for i in predictions] + [predictions] + predictions2
            except:
                predictions = [i if type(i)==list else [i] for i in predictions] + [predictions]
        # print(predictions)
    else:
        predictions = [[]]
    return predictions


def re_extract_df_string(string):
    if len(string) == 0:
        return []
    list_string = re.split(r'[\'\n\",;\(\)\[\]]', string)
    list_string = [i for i in list_string if len(''.join(i.split(' '))) > 0]
    return list_string


def obtain_answer_df(result):
    answers = [re_extract_df_string(result['ans{}'.format(i)]) for i in range(1, 4) if result['ans{}'.format(i)]]
    return answers


def obtain_prediction_df(prediction_text):
    predictions = [re_extract_df_string(prediction_text)]
    return predictions


def obtain_answer_str(result):
    answers = [[result['ans{}'.format(idx)]] for idx in range(1, 4) if result['ans{}'.format(idx)]]
    return answers


def obtain_prediction_str(prediction_text):
    predictions = re.split(r'[\'\n\",;\(\)\[\]]', prediction_text)
    predictions = [[i] for i in predictions if len(''.join(i.split(' '))) > 0]
    return predictions



def match_alltype_new(predictions, answers):
    # predictions: a list of list
    # answers: a list of list
    full_correct, partial_correct, f1 = False, False, 0.0
    try:
        full_correct = any([pred_list in answers for pred_list in predictions])
    except:
        # print()
        pass
    partial_correct = full_correct
    for pred_list in predictions:
        if len(pred_list) > 0:
            if any([item in ans_list for item in pred_list for ans_list in answers]):
                partial_correct = True
                break
    try:
        f1 = max([compute_macro_f1(answers, pred_list) for pred_list in predictions])
    except:
        pass
    # f1 = max([compute_macro_f1(predictions, ans_list) for ans_list in answers])
    return {'f1': f1,
            'correct': full_correct,
            'partial_correct': partial_correct}


def compare_answers(this_index):
    assert this_index['nbid'] == this_index['nbid']
    result = {'idx': this_index['nbid'],
              'answer_type': this_index['answer_type'],
              'error': False,
              'strict_correct': False,
              'correct': False,
              'partial_correct': False,
              'f1': 0.0,
              'answers': [],
              'predictions': [],}
    answers, predictions = [], []
    if this_index['output_type'] == 'error':
        # if have error (exception), then not correct
        result['error'] = True
    elif this_index['output_type'] == 'stream' or this_index['output_type'] == 'execute_result':
        if len(this_index['text']) != 0:
            # if not have textual output, then not correct
            execution = ''.join(this_index['text'])
            if strict_correct_match(execution, this_index)['strict_correct']:
                result.update(strict_correct_match(execution, this_index))
            else:
                result['strict_correct'] = False
                if this_index['answer_type'] == 'number':
                    answers = obtain_answer_number(this_index)
                    predictions = obtain_prediction_number(execution)
                    result.update(match_alltype_new(predictions, answers))
                elif 'array' in this_index['answer_type']:
                    answers = obtain_answer_array(this_index)
                    predictions = obtain_prediction_array(execution, this_index['answer_type'])
                    result.update(match_alltype_new(predictions, answers))
                elif this_index['answer_type'] == 'df':
                    answers = obtain_answer_df(this_index)
                    predictions = obtain_prediction_df(execution)
                    result.update(match_alltype_new(predictions, answers))
                elif this_index['answer_type'] == 'str':
                    answers = obtain_answer_str(this_index)
                    predictions = obtain_prediction_str(execution)
                    result.update(match_alltype_new(predictions, answers))
    if result['answers'] == [] and result['predictions'] == []:
        result['answers'] = answers
        result['predictions'] = predictions
    return result


def error_statistic(indexes):
    error_counter = Counter()
    for this_index in indexes:
        if this_index['text']:
            error_name = this_index['text'].split(': ')[0]
            if error_name == 'NameError':
                error_counter['VarAPINotDefined'] += 1
            elif error_name == 'SyntaxError':
                error_counter['InvalidSyntax'] += 1
        else:
            error_counter['NoMessage'] += 1


def error_count(indexes):
    error_counter = Counter()
    for this_index in indexes:
        if this_index['text'] and type(this_index['text'])==str:
            error_name = this_index['text'].split(': ')[0]
            error_counter[error_name] += 1
        else:
            error_counter['NoMessage'] += 1
    return error_counter

def write_one_notebook(args, path_in, path_out, nbid, row_id, replace_code):
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    files = os.listdir(path_in)
    for file in files:
        if file.endswith('.ipynb'):
            notebook = read_json(os.path.join(path_in, file))
            new_cell = copy.deepcopy(notebook['cells'][int(row_id)])
            ori_code = ''.join(new_cell['source'])
            # replace ori code to replace code
            if not compare_code_string(ori_code, replace_code):
                print('Cell or code mismatch: {} {}'.format(nbid, row_id))
                # print('ori_code: {}'.format(ori_code))
                # print('replace_code: {}'.format(replace_code))
            new_cell['source'] = replace_code.split('\n')
            new_cell['source'][:-1] = [item + '\n' for item in new_cell['source'][:-1]]
            notebook['cells'][int(row_id)] = new_cell
            # replace the path in the cells to new path
            notebook['cells'] = replace_base_path(notebook['cells'], args.data_dir, args.path_save_notebooks)
            notebook['cells'] = replace_file_dir_path(notebook['cells'], nbid, nbid+'R{}'.format(row_id))
            notebook['cells'] = replace_file_dir_path(notebook['cells'], './ok_notebooks_rerun_january', args.path_save_notebooks)
            # write_json(notebook, os.path.join(path_out, file))
            write_json(notebook, os.path.join(path_out, nbid+'R{}.ipynb'.format(row_id)))
        elif 'ipynb_checkpoints' in file:
            continue
        else:
            if platform.system().lower() == 'windows':
                # pass
                cmd = 'copy \"{}\" {}'.format(os.path.join(path_in, file), path_out).replace('/', '\\')
                os.system(cmd)
            elif platform.system().lower() == 'darwin' or platform.system().lower() == 'linux':
                # pass
                os.system('cp \"{}\" {}'.format(os.path.join(path_in, file), path_out))
            else:
                print("Error system: {}".format(platform.system().lower()))


def process_args(args):
    args.path_index = os.path.join(args.path_save_notebooks, args.path_index)
    args.path_csv = os.path.join(args.data_dir, args.path_csv)
    if not os.path.exists(args.path_save_notebooks):
        os.makedirs(args.path_save_notebooks)
    return args


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='../dataset/ExeDS_notebooks', type=str,
                    help="The path to notebooks and raw data.")
parser.add_argument("--path_dataset", default="../dataset/exeds_test.json",
                    type=str, help="The path to processed dataset.")
parser.add_argument("--path_generation", default="../saved_models/pymt5/split_generation_results.json",
                    type=str, help="The path to the generated code.")
parser.add_argument("--path_save_notebooks", default="../saved_models/pymt5/testbed_notebooks",
                    type=str, help="The path to save notebooks.")
parser.add_argument("--path_index", default="index.json", type=str,
                    help="The path to index json file .")
parser.add_argument("--path_csv", default="answers.csv", type=str,
                    help="The path to csv .")

parser.add_argument("--do_create_notebook", action='store_true', help="Whether to run write generation.")
parser.add_argument("--do_run", action='store_true', help="Whether to run notebooks.")
parser.add_argument("--do_evaluate", action='store_true', default=True, help="Whether to run evaluation.")

args = parser.parse_args()
args = process_args(args)

# read generations
generations = read_json(args.path_generation)
generations = [{"target": decode_string(item["target"]),
                "generation": decode_string(item["generation"]), } for item in generations]
dataset = read_json(args.path_dataset)
dataset_files = [{'idx': idx, 'file': item['file'], 'row_id':item['output']['position'], 'code':item['output']['ori']} for idx, item in enumerate(dataset)]
print("length of dataset_files: {}".format(len(dataset_files)))
# read answers
csv_file = csv_reader(args.path_csv)
header = csv_file[0]
csv_file = csv_file[1:]
csv_file = {item[0].split('.')[0] + 'R{}'.format(item[1]): item for item in csv_file}

# process data, create new notebooks and save index files
if args.do_create_notebook:
    indexes = []
    # generation_idx = 0
    # for dataset_file in dataset_files:
    #     src_code = ''.join(dataset_file['code'].split(' '))
    #     try:
    #         generation_code = merge_generation_code(generations[generation_idx]['generation'])
    #     except IndexError:
    #         continue
    #     # generation_code = merge_generation_code(generations[generation_idx]['target'])
    #     if ''.join(dataset_file['code'].split(' ')) != ''.join(generations[generation_idx]['target'].split(' ')):
    #         print(generation_idx)
    #         print(''.join(dataset_file['code'].split(' ')))
    #         print(''.join(generations[generation_idx]['target'].split(' ')))
    #         if 'norm.fit(data)' in ''.join(dataset_file['code'].split(' ')) \
    #                 and 'norm.fit(data)' in ''.join(generations[generation_idx]['target'].split(' ')):
    #             generation_idx += 1
    #             print(generation_idx)
    #         continue
    #     path_in = os.path.join(args.data_dir, dataset_file['file'])
    #     path_out = os.path.join(args.path_save_notebooks, dataset_file['file']+'R{}'.format(dataset_file['row_id']))
    #     write_one_notebook(args, path_in, path_out, dataset_file['file'], dataset_file['row_id'], generation_code)
    #     generation_idx += 1
    generation_map = {''.join(generations[idx]['target'].split(' ')): idx for idx in range(len(generations))}
    non_map_items = 0
    for idx, dataset_file in enumerate(dataset_files):
        src_code = ''.join(dataset_file['code'].split(' '))
        if src_code in generation_map:
            generation_item = generations[generation_map[src_code]]
            dataset_files[idx]['generation'] = merge_generation_code(generation_item['generation'])
        else:
            dataset_files[idx]['generation'] = ""
            non_map_items+=1
    print("we have {} valid generations: ".format(len(dataset_files)-non_map_items))

    for dataset_file in dataset_files:
        generation_code = dataset_file['generation']
        path_in = os.path.join(args.data_dir, dataset_file['file'])
        path_out = os.path.join(args.path_save_notebooks, dataset_file['file']+'R{}'.format(dataset_file['row_id']))
        write_one_notebook(args, path_in, path_out, dataset_file['file'], dataset_file['row_id'], generation_code)

        item_csv = csv_file[dataset_file['file'] + 'R{}'.format(dataset_file['row_id'])]
        this_index = {'idx': len(indexes),
                      'dir': dataset_file['file']+'R{}'.format(dataset_file['row_id']),
                      'nbid': dataset_file['file']+'R{}'.format(dataset_file['row_id']),
                      'row_id': dataset_file['row_id'],
                      'answer_type': item_csv[9],
                      'ans1': item_csv[2], 'ans2': item_csv[3], 'ans3': item_csv[4],
                      'code1': item_csv[5], 'code2': item_csv[6], 'code3': item_csv[7],
                      }
        indexes.append(this_index)
    print("length of indexes: {}".format(len(indexes)))
    write_json(indexes, args.path_index)

# running evaluation
if args.do_run:
    print("Start running notebooks")
    time.sleep(5)
    t1 = time.time()
    cmd = f'python run_one_nb_rerun.py --base_dir {args.path_save_notebooks} --idx_path {os.path.basename(args.path_index)} 2>&1 |tee {args.path_save_notebooks}/replace_base.log'
    print("cmd: {}".format(cmd))
    os.system(cmd)
    print("time: {}s".format(time.time()-t1))

# test
if args.do_evaluate:
    # indexes = read_json(args.path_index)
    indexes = []
    # generation_idx = 0
    # for dataset_file in dataset_files:
    #     try:
    #         if ''.join(dataset_file['code'].split(' ')) != ''.join(generations[generation_idx]['target'].split(' ')):
    #             print(generation_idx)
    #             print(''.join(dataset_file['code'].split(' ')))
    #             print(''.join(generations[generation_idx]['target'].split(' ')))
    #             continue
    #     except IndexError:
    #         continue
    #     generation_idx += 1
    generation_map = {''.join(generations[idx]['target'].split(' ')): idx for idx in range(len(generations))}
    non_map_items = 0
    for idx, dataset_file in enumerate(dataset_files):
        src_code = ''.join(dataset_file['code'].split(' '))
        if src_code in generation_map:
            generation_item = generations[generation_map[src_code]]
            dataset_files[idx]['generation'] = merge_generation_code(generation_item['generation'])
        else:
            dataset_files[idx]['generation'] = ""
            non_map_items+=1
    print("we have {} valid generations: ".format(len(dataset_files)-non_map_items))
    for dataset_file in dataset_files:
        item_csv = csv_file[dataset_file['file'] + 'R{}'.format(dataset_file['row_id'])]
        this_index = {'idx': len(indexes),
                      'dir': dataset_file['file']+'R{}'.format(dataset_file['row_id']),
                      'nbid': dataset_file['file']+'R{}'.format(dataset_file['row_id']),
                      'row_id': dataset_file['row_id'],
                      'answer_type': item_csv[9],
                      'ans1': item_csv[2], 'ans2': item_csv[3], 'ans3': item_csv[4],
                      'code1': item_csv[5], 'code2': item_csv[6], 'code3': item_csv[7],
                      }
        indexes.append(this_index)
    print("length of indexes: {}".format(len(indexes)))
    write_json(indexes, args.path_index)

    all_output = []
    all_results = []
    print("length of indexes: {}".format(len(indexes)))
    for this_index in indexes:
        # print(os.path.join(args.path_save_notebooks, this_index['dir'], 'run_{}.ipynb'.format(this_index['dir'])))
        notebook = read_json(os.path.join(args.path_save_notebooks, this_index['dir'], 'run_{}.ipynb'.format(this_index['dir'])))
        # if this_index['dir'] == 'file_126993R11':
        output = obtain_cell_output(notebook, this_index)
        this_index.update(output)
        all_output.append(this_index)
        all_results.append(compare_answers(this_index))
    results_errors = [this_index for this_index in all_output if this_index['output_type'] == 'error']
    error_counter = error_count(results_errors)
    print("### Execution Accuracy: \nCorrect\tF1 \tNoErrRate")
    print("{}  \t{}  \t{}".format(round(100*sum([i['correct'] for i in all_results])/len(all_results), 2),
                                  round(100*sum([i['f1'] for i in all_results])/len(all_results), 2),
                                  round(100*(len(all_output)-len(results_errors))/len(all_output),2),))
    print(error_counter)
    write_json(all_output, os.path.join(args.path_save_notebooks, 'temp_output.json'))
    write_json(all_results, os.path.join(args.path_save_notebooks, 'eval_results.json'))


