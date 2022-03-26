import json
import os
import argparse
import re
import copy
import random
import tokenizers
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from usage_pattern import ONE_KEY_SIMPLE_PATTERNS, ONE_KEY_MULTIPLE_PATTERNS, ONE_KEY_CALL_PATTERNS, TWO_KEY_CALL_PATTERNS, TWO_KEY_SIMPLE_PATTERNS
from usage_pattern import TOKENS_ONE_KEY_SIMPLE_PATTERNS, TOKENS_ONE_KEY_MULTIPLE_PATTERNS, TOKENS_ONE_KEY_CALL_PATTERNS, TOKENS_TWO_KEY_CALL_PATTERNS, TOKENS_TWO_KEY_SIMPLE_PATTERNS


def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def write_jsonl(data, path):
    with open(path, 'w') as fp:
        for inst in data:
            fp.write(json.dumps(inst) + '\n')


def read_jsonl(path):
    data = []
    with open(path, 'r') as fp:
        for line in fp:
            line = line.strip()
            data.append(json.loads(line))
    return data


def get_c_list(column_list):
    c_list = set()
    for i in column_list:
        try:
            str_i = i
            if i[0] == '\'' or i[0] == '\"':
                str_i = i[1:-1]
            if len(str_i) > 0:
                c_list.add('\'' + str_i + '\'')
        except:
            continue
    c_list = list(c_list)
    return c_list


def get_demonstration(column_list, max_columns, output_num=7, madeupword_flag=False, token=True):
    column_list = get_c_list(column_list)
    column_list = column_list[:max_columns]
    demos, output_demos = [], []
    if not token:
        one_key_simple_patterns, one_key_multiple_patterns, one_key_call_patterns, two_key_call_patterns, two_key_simple_patterns = ONE_KEY_SIMPLE_PATTERNS, ONE_KEY_MULTIPLE_PATTERNS, ONE_KEY_CALL_PATTERNS, TWO_KEY_CALL_PATTERNS, TWO_KEY_SIMPLE_PATTERNS
    else:
        one_key_simple_patterns, one_key_multiple_patterns, one_key_call_patterns, two_key_call_patterns, two_key_simple_patterns = TOKENS_ONE_KEY_SIMPLE_PATTERNS, TOKENS_ONE_KEY_MULTIPLE_PATTERNS, TOKENS_ONE_KEY_CALL_PATTERNS, TOKENS_TWO_KEY_CALL_PATTERNS, TOKENS_TWO_KEY_SIMPLE_PATTERNS

    if len(column_list) > 0:
        # 5 != max_columns
        used_columns = random.sample(column_list, 5) if len(column_list) > 5 else column_list[:5]
        call_used_columns = [c[1:-1] if c[0]=="\'" or c[0]=="\"" else c for c in used_columns]
        simple_used_columns = copy.deepcopy(used_columns)
        if madeupword_flag:
            column2madeupid = dict(zip(column_list, [i + 87 for i in range(len(column_list))]))
            simple_used_columns = ["{}{}".format(item, column2madeupid[item]) for item in simple_used_columns]
        pad_idx = np.random.choice(range(len(simple_used_columns)), 5-len(simple_used_columns)).tolist()
        simple_used_columns = simple_used_columns + [simple_used_columns[i] for i in pad_idx]
        call_used_columns = call_used_columns + [call_used_columns[i] for i in pad_idx]
        demos.append(random.sample(one_key_simple_patterns, 1)[0].format(simple_used_columns[0]))
        demos.append(random.sample(one_key_simple_patterns, 1)[0].format(simple_used_columns[1]))
        demos.append(random.sample(one_key_multiple_patterns, 1)[0].format(simple_used_columns[2], simple_used_columns[2]))
        demos.append(random.sample(two_key_simple_patterns, 1)[0].format(simple_used_columns[3], simple_used_columns[4]))
        demos.append(random.sample(two_key_simple_patterns, 1)[0].format(simple_used_columns[4], simple_used_columns[3]))
        demos.append(random.sample(one_key_call_patterns, 1)[0].format(call_used_columns[2]))
        demos.append(random.sample(two_key_call_patterns, 1)[0].format(call_used_columns[0], call_used_columns[4]))

        random.shuffle(demos) # list
        sample_num = min(output_num, len(demos))
        output_demos = random.sample(demos, sample_num)

        if token:
            output_demos = [item2 for item in output_demos for item2 in item.split(' ')]

    return ['#', 'CODE', ':',] + output_demos


def get_demonstration_output(column_list, max_columns, madeupword_flag=False, token=True):
    column_list = get_c_list(column_list)
    column_list = column_list[:max_columns]
    demos, output_demos = [], []
    if token:
        one_key_simple_patterns, one_key_multiple_patterns, one_key_call_patterns, two_key_call_patterns, two_key_simple_patterns = ONE_KEY_SIMPLE_PATTERNS, ONE_KEY_MULTIPLE_PATTERNS, ONE_KEY_CALL_PATTERNS, TWO_KEY_CALL_PATTERNS, TWO_KEY_SIMPLE_PATTERNS
    else:
        one_key_simple_patterns, one_key_multiple_patterns, one_key_call_patterns, two_key_call_patterns, two_key_simple_patterns = TOKENS_ONE_KEY_SIMPLE_PATTERNS, TOKENS_ONE_KEY_MULTIPLE_PATTERNS, TOKENS_ONE_KEY_CALL_PATTERNS, TOKENS_TWO_KEY_CALL_PATTERNS, TOKENS_TWO_KEY_SIMPLE_PATTERNS

    if len(column_list) > 0:
        # 2 != max_columns
        used_columns = random.sample(column_list, 2) if len(column_list) > 2 else column_list[:2]
        call_used_columns = [c[1:-1] if c[0]=="\'" or c[0]=="\"" else c for c in used_columns]
        simple_used_columns = copy.deepcopy(used_columns)
        if madeupword_flag:
            column2madeupid = dict(zip(column_list, [i + 87 for i in range(len(column_list))]))
            simple_used_columns = ["{}{}".format(item, column2madeupid[item]) for item in simple_used_columns]
        pad_idx = np.random.choice(range(len(simple_used_columns)), 2-len(simple_used_columns)).tolist()
        simple_used_columns = simple_used_columns + [simple_used_columns[i] for i in pad_idx]
        call_used_columns = call_used_columns + [call_used_columns[i] for i in pad_idx]
        demos.append(random.sample(one_key_simple_patterns, 1)[0].format(simple_used_columns[0]))
        demos.append(random.sample(one_key_simple_patterns, 1)[0].format(simple_used_columns[1]))
        demos.append(random.sample(one_key_multiple_patterns, 1)[0].format(simple_used_columns[2], simple_used_columns[2]))
        demos.append(random.sample(two_key_simple_patterns, 1)[0].format(simple_used_columns[3], simple_used_columns[4]))
        demos.append(random.sample(two_key_simple_patterns, 1)[0].format(simple_used_columns[4], simple_used_columns[3]))
        demos.append(random.sample(one_key_call_patterns, 1)[0].format(call_used_columns[2]))
        demos.append(random.sample(two_key_call_patterns, 1)[0].format(call_used_columns[0], call_used_columns[4]))
        output_demos = random.sample(demos, 2)
        if token:
            output_demos = [item2 for item in output_demos for item2 in item.split(' ')]

    return ['#', 'CODE', ':',] + output_demos


def get_column_input(column_list, max_columns, madeupword_flag=False):
    column_list = get_c_list(column_list)
    column_list = column_list[:max_columns]
    column2madeupid = dict(zip(column_list, [i + 87 for i in range(len(column_list))]))
    col_toks = ['#', 'TABLE', '\n', 'df.columns = ']
    madeup_toks = []
    if len(column_list) == 1:
        col_toks.append('[' + column_list[-1] + ']\n')
        if madeupword_flag:
            madeup_toks.append(column_list[-1])
            madeup_toks.append('madeupword' + str(87).zfill(4) + '\n')
            col_toks.extend(madeup_toks)
        return col_toks

    for i, col in zip(range(len(column_list) - 1), column_list[:-1]):
        if i == 0:
            col_toks.append('[' + col + ',')
        else:
            col_toks.append(col + ',')
        madeup_toks.append(col)
        if madeupword_flag:
            madeup_toks.append('madeupword' + str(i + 87).zfill(4))
    col_toks.append(column_list[-1] + ']\n')
    if madeupword_flag:
        madeup_toks.append(column_list[-1])
        madeup_toks.append('madeupword' + str(len(column_list) + 2).zfill(4) + '\n')
        col_toks.extend(madeup_toks)
    return col_toks


def get_column_code_input(column_list, code, max_columns, madeupword_flag=False):
    if madeupword_flag:
        column_list = get_c_list(column_list)
        column_list = column_list[:max_columns]
        column2madeupid = dict(zip(column_list, [i + 87 for i in range(len(column_list))]))
        for col in column_list:
            code = code.replace(col, col + ' madeupword' + str(column2madeupid[col]).zfill(4))
    return code


def get_column_code_input_mask_tab(column_list, code, max_columns, madeupword_flag=False):
    if madeupword_flag:
        column_list = get_c_list(column_list)
        # column_list = column_list[:max_columns]
        for col in column_list:
            code = code.replace(col, 'madeupword0002')
    return code


def cleaning_string(string):
    if string:
        string = re.sub(r'\n[ \n\t]*\n', r'\n', string)  # remove extra \n\n
        string = re.sub("\"", "\'", string)
        return string
    else:
        return ''

def get_simple_input_output(item, args):
    # clean the data first
    item['output']['str'] = cleaning_string(item['output'].get('str', ''))
    for i in range(1, args.context_range+1):
        item['context_dist{}'.format(i)]['str'] = cleaning_string(item['context_dist{}'.format(i)].get('str', ''))
    item['table_keys'] = [cleaning_string(tab_key) for tab_key in item['table_keys']]

    # table
    if not args.not_add_table:
        column_list = item['table_keys']
        if len(column_list) != 0:
            table_tokens_pymt5 = get_column_input(column_list, args.max_columns, args.madeupword_flag)
            table_tokens_gpt = get_column_input(column_list, args.max_columns, False)
            if args.table_prompt == 'df-usage' or args.table_prompt == 'df-usage2':
                if args.token_type == 'str':
                    table_tokens_pymt5.extend(get_demonstration(column_list, args.max_columns, 7, args.madeupword_flag, False))
                    table_tokens_gpt.extend(get_demonstration(column_list, args.max_columns, 7, args.madeupword_flag, False))
                elif args.token_type == 'token':
                    table_tokens_pymt5.extend(get_demonstration(column_list, args.max_columns, 7, False, True))
                    table_tokens_gpt.extend(get_demonstration(column_list, args.max_columns, 7, False, True))
                else:
                    pass
        else:
            table_tokens_pymt5 = []
            table_tokens_gpt = []
        table_str_pymt5 = ' '.join(table_tokens_pymt5)
        table_str_pymt5 = re.sub(' +', ' ', table_str_pymt5)  # remove extra space
        table_str_gpt = ' '.join(table_tokens_gpt)
        table_str_gpt = re.sub(' +', ' ', table_str_gpt)  # remove extra space

    # context
    # contexts = [item['context_dist{}'.format(i)] for i in reversed(range(1, args.context_range+1)) if item['context_dist{}'.format(i)] and item['context_dist{}'.format(i)]['str'] and item['context_dist{}'.format(i)]['token']]
    if 'nonl' in args.split:
        contexts = [item['context_dist{}'.format(i)] for i in reversed(range(2, args.context_range + 1)) if
                    item['context_dist{}'.format(i)].get('ori') and item['context_dist{}'.format(i)].get('ori_token')]
    else:
        contexts = [item['context_dist{}'.format(i)] for i in reversed(range(1, args.context_range + 1)) if
                    item['context_dist{}'.format(i)].get('ori') and item['context_dist{}'.format(i)].get('ori_token')]
    # print(len(contexts))
    # print(contexts)
    inputs = []
    for context_i in contexts:
        if context_i['cell_type'] == 'code':
            if args.token_type == 'str':
                code_toks = context_i['ori'].split(' ')[:args.max_code_cell_tokens]
            else:
                code_toks = context_i['ori_token'][:args.max_code_cell_tokens]
            inputs.extend(['#', 'CODE', '\n'] + code_toks)
        else:
            # markdown
            inputs.extend(['#', 'MARKDOWN', '\n'] + context_i['ori_token'][:args.max_md_cell_tokens])
    inputs_str = ' '.join(inputs)
    inputs_str = inputs_str.replace('\'jupyter_string\'', 'madeupword0001')
    inputs_str = inputs_str.replace('jupyter_string', 'madeupword0001')

    # code
    column_list = item['table_keys']
    if len(column_list) != 0:
        token_type_input = item['output']['ori'] if args.token_type == 'str' else ' '.join(item['output']['ori_token'])
        code_pymt5 = get_column_code_input(column_list, token_type_input, args.max_columns, args.madeupword_flag)
        code_gpt = token_type_input
        # if args.mask_type == 'ori':
        #     token_type_input = item['output']['ori'] if args.token_type == 'str' else ' '.join(item['output']['ori_token'])
        #     code_pymt5 = get_column_code_input(column_list, token_type_input, args.max_columns, args.madeupword_flag)
        #     code_gpt = token_type_input
        # elif args.mask_type == 'jupyter':
        #     token_type_input = item['output']['str'] if args.token_type == 'str' else ' '.join(item['output']['token'])
        #     code_pymt5 = get_column_code_input(column_list, token_type_input, args.max_columns, args.madeupword_flag)
        #     code_gpt = token_type_input
        # elif args.mask_type == 'jutab':
        #     token_type_input = item['output']['str'] if args.token_type == 'str' else ' '.join(item['output']['token'])
        #     code_pymt5 = get_column_code_input_mask_tab(column_list, token_type_input, args.max_columns, args.madeupword_flag)
        #     code_gpt = token_type_input
        # else:
        #     raise Exception(" wrong args.mask_type in [ori, jupyter, julab]")
        if args.table_prompt == 'df-usage2' and args.split == 'train':
            add_tokens = get_demonstration(column_list, args.max_columns, 2, args.madeupword_flag, token=False)[3:]
            code_pymt5 = ' '.join(add_tokens) + code_pymt5
            add_tokens_gpt = get_demonstration(column_list, args.max_columns, 2, False, token=False)[3:]
            code_gpt = ' '.join(add_tokens_gpt) + code_gpt
    else:
        code_pymt5 = item['output']['ori'] if args.token_type == 'str' else ' '.join(item['output']['ori_token'])
        code_gpt = code_pymt5
        # if args.mask_type == 'ori':
        #     code_pymt5 = item['output']['ori'] if args.token_type == 'str' else ' '.join(item['output']['ori_token'])
        #     code_gpt = code_pymt5
        # else:
        #     code_pymt5 = item['output']['str'] if args.token_type == 'str' else ' '.join(item['output']['token'])
        #     code_gpt = code_pymt5
    code_pymt5 = re.sub(r'\n[ \n\t]*\n', r'\n', code_pymt5)  # remove extra \n\n
    code_pymt5 = re.sub(' +', ' ', code_pymt5)  # remove extra space
    code_pymt5 = code_pymt5.replace('\'jupyter_string\'', 'madeupword0001')
    code_pymt5 = code_pymt5.replace('jupyter_string', 'madeupword0001')
    code_gpt = re.sub(r'\n[ \n\t]*\n', r'\n', code_gpt)  # remove extra \n\n
    code_gpt = re.sub(' +', ' ', code_gpt)  # remove extra space

    if not args.not_add_table:
        ctx_pymt5 = table_str_pymt5 + ' ' + inputs_str
        ctx_gpt = table_str_gpt + ' ' + inputs_str
    else:
        ctx_pymt5 = inputs_str
        ctx_gpt = inputs_str
    return {'idx': item['idx'], 'url': item['url'], 'ctx_pymt5': ctx_pymt5, 'ctx': ctx_gpt, 'code_pymt5': code_pymt5, 'code': code_gpt, 'executable': item['executable']}


def remove_abnormal_examples(train_1):
    train = []
    for item in train_1:
        add_sample = False
        ctx_len = len(item['ctx'].split())
        code_len = len(item['code'].split())
        # if nl_len == 0 or code_len == 0:
        #     continue
        if ctx_len + code_len <= args.max_ctx_cell_tokens and code_len <= args.max_code_cell_tokens:
            num_line = len(re.split(r'\n+', item['code']))
            if num_line <= args.max_line and num_line >= args.min_line:
                add_sample = True
        if args.only_executable and item['executable'] != 1:
            add_sample = False
        if args.only_not_executable and item['executable'] == 1:
            add_sample = False

        if add_sample:
            train.append(item)
    return train


def item_split(item):
    item_list = re.split(r'(madeupword\d{4})', item)
    item_span = []
    token = []
    sep_list = []
    for s in item_list:
        if 'madeupword' in s:
            token.append(' '.join(item_span))
            item_span = []
            sep_list.append(s)
        else:
            item_span.append(s)
    token.append(' '.join(item_span))
    sep_list.append('EMPTY')
    return token, sep_list


def get_tokenize(nl_whole):
    token, sep_list = item_split(nl_whole)
    encode_src = tokenizer.encode_batch([s for s in token])
    token_list = []
    for s in encode_src:
        token_list.append(s.tokens)
    return token_list


def get_string(token_list, sep_list):
    final_s = []
    for s, sep in zip(token_list, sep_list):
        final_s.extend(s)
        if sep != 'EMPTY':
            final_s.append(sep)
    return final_s


def get_whole_tokenize(nl):
    token, sep_list = item_split(nl)
    token_list = get_tokenize(nl)
    return get_string(token_list, sep_list)


def get_gpt_neo_format(item, split):
    gpt_neo_item = {}
    text = "# Context is: \n"
    text += item['ctx']
    text += "\n # Code of the context is: \n"
    if split != 'test':
        text += item['code']
    gpt_neo_item['text'] = text
    gpt_neo_item['code'] = item['code']
    gpt_neo_item['idx'] = item['idx']
    gpt_neo_item['url'] = item['url']
    gpt_neo_item['executable'] = item['executable']
    return gpt_neo_item


def process_args(args):
    args.max_line = int(args.line_length.split('-')[1])
    args.min_line = int(args.line_length.split('-')[0])
    args.file_name = os.path.join(args.basic_dir, 'dataset', args.file_name)

    save_dir = 'prepro'
    if args.not_add_table:
        save_dir += '_noTab'
    else:
        save_dir += '_addTab'
        save_dir += '-{}'.format(args.table_prompt)
    if args.madeupword_flag:
        save_dir += '_madeup'
    # save_dir += '_{}'.format(args.mask_type)
    if args.only_executable:
        save_dir += '_onlyexecutable'
    if args.only_not_executable:
        save_dir += '_onlynotexecutable'
    save_dir += '_{}'.format(args.token_type)
    save_dir += '_range{}'.format(args.context_range)
    save_dir += '_lineLen{}'.format(args.line_length)
    save_dir += '_c{}m{}a{}'.format(args.max_code_cell_tokens, args.max_md_cell_tokens, args.max_ctx_cell_tokens)
    args.save_dir = os.path.join(args.basic_dir, 'preprocessed_data', save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.save_file_name = os.path.join(args.save_dir, '{}.json'.format(args.split))
    args.save_tokenization_prefix = os.path.join(args.save_dir, 'fairseq_tokenization')
    if not os.path.exists(args.save_tokenization_prefix):
        os.makedirs(args.save_tokenization_prefix)
    args.save_gpt_neo_prefix = os.path.join(args.save_dir, 'gpt_neo')
    if not os.path.exists(args.save_gpt_neo_prefix):
        os.makedirs(args.save_gpt_neo_prefix)

    print("save files to {}".format(args.save_dir))
    print("save tokenization to {}".format(args.save_tokenization_prefix))
    print("save gptneo to {}".format(args.save_gpt_neo_prefix))
    print("save {} file to {}".format(args.split, args.save_file_name))
    return args


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--split', default='test', type=str, choices=['train', 'dev', 'test'])
    parser.add_argument('--split', default='test', type=str, choices=['train', 'dev', 'test', 'dev2k', 'dev-nonl', 'test-nonl'])
    parser.add_argument('--not_add_table', action='store_true')
    parser.add_argument('--madeupword_flag', action='store_true', default=True)
    parser.add_argument('--only_executable', action='store_true')
    parser.add_argument('--only_not_executable', action='store_true', help='only_executable and only_not_executable can not use together ')
    parser.add_argument('--do_fairseq_tokenization', action='store_true')
    parser.add_argument('--do_gptneo', action='store_true')
    parser.add_argument('--reprocess', action='store_true')

    parser.add_argument("--basic_dir", default='../', type=str, help="basic dir for save preprocessed data.")
    parser.add_argument("--file_name", default='exeds_dev.json', type=str, help="file name for data to preprocess.")
    parser.add_argument('--table_prompt', default='df', type=str, choices=['df', 'df-usage', 'df-usage2'])
    parser.add_argument('--token_type', default='token', type=str, choices=['token', 'str'])
    # parser.add_argument('--mask_type', default='jupyter', type=str, choices=['ori', 'jupyter', 'jutab'])
    parser.add_argument('--save_dir', default='', type=str, )
    parser.add_argument('--save_file_name', default='', type=str, )
    # parser.add_argument("--dir_output_split",
    #                     default='preprocessed_data/juice/',
    #                     type=str, help="The output directory where the examples met the requirement of GPT length")

    parser.add_argument("--max_columns", default=30, type=int,
                        help="Maximum number of columns to be considered in a table.")
    parser.add_argument("--context_range", default=3, type=int, help="context cell range")

    parser.add_argument("--max_code_cell_tokens", default=200, type=int,
                        help="Number of tokens in each code cell to use above")
    parser.add_argument("--max_md_cell_tokens", default=200, type=int,
                        help="Number of tokens in each markdown cell to use above")
    parser.add_argument("--max_ctx_cell_tokens", default=900, type=int,
                        help="Number of tokens in each context to use above")
    parser.add_argument("--line_length", default='1-25', type=str, help="Number of lines to be considered in code.")
    parser.add_argument("--max_line", default=0, type=int, help="Maximum number of lines to be considered in code.")
    parser.add_argument("--min_line", default=0, type=int, help="Minimum number of lines to be considered in code.")

    # tokenization
    parser.add_argument("--tok_prefix",
                        default='../jupyt5_weights/roberta_aug_spaces',
                        type=str, help="The tokenizer's location")
    parser.add_argument("--lang", default='python', type=str, help="Coding language type")

    args = parser.parse_args()
    args = process_args(args)

    if os.path.exists(args.save_file_name) and not args.reprocess:
        raise Exception("args.save_file_name {} exists, call --reprocess to reprocess".format(args.save_file_name))
    else:
        input_data = read_json(args.file_name)
        print("number of input data: {}".format(len(input_data)))

        n_threads = 24
        with Pool(n_threads) as p:
            func_ = partial(get_simple_input_output, args=args, )
            all_results = list(
                tqdm(p.imap(func_, input_data, chunksize=16), total=len(input_data), desc="preprocessing input output", ))
            print("number of output data before remove_abnormal_examples: {}".format(len(all_results)))
            all_results = remove_abnormal_examples(all_results)
            write_json(all_results, args.save_file_name)
            print("number of output data after remove_abnormal_examples: {}\n".format(len(all_results)))

        # all_results = []
        # for idx, item in enumerate(input_data):
        # # for idx, item in enumerate(input_data[3:]):
        #     # print(idx)
        #     all_results.append(get_simple_input_output(item, args=args))

    if args.do_fairseq_tokenization:
        print('Processing {} from {}'.format(args.split, args.save_file_name))
        all_results = read_json(args.save_file_name)
        tokenizer = tokenizers.ByteLevelBPETokenizer(args.tok_prefix + "-vocab.json", args.tok_prefix + "-merges.txt")

        nl = [item['ctx_pymt5'] for item in all_results]
        code = [item['code_pymt5'] for item in all_results]
        with open(os.path.join(args.save_tokenization_prefix, f'{args.lang}.{args.split}_nl_to_code.tgt'),
                  'w', encoding='utf-8') as f_tgt:
            for code_item in tqdm(code, desc='fairseq tokenization for code'):
                # code_item = code_item.replace('\"jupyter_string\"', 'madeupword0001')
                # code_item = code_item.replace('\'jupyter_string\'', 'madeupword0001')
                t_string = get_whole_tokenize(code_item)
                f_tgt.write(' '.join(t_string) + '\n')

        with open(os.path.join(args.save_tokenization_prefix, f'{args.lang}.{args.split}_nl_to_code.src'),
                  'w', encoding='utf-8') as f_src:
            for nl_item in tqdm(nl, desc='fairseq tokenization for context'):
                # nl_item = nl_item.replace('\"jupyter_string\"', 'madeupword0001')
                # nl_item = nl_item.replace('\'jupyter_string\'', 'madeupword0001')
                t_string = get_whole_tokenize(nl_item)
                f_src.write(' '.join(t_string) + '\n')

    if args.do_gptneo:
        traindev = args.split.split('_')[0]
        print('Processing {} ({}) from {}'.format(args.split, traindev, args.save_file_name))
        all_results = read_json(args.save_file_name)
        gpt_neo_item = [get_gpt_neo_format(item, traindev) for item in all_results]

        mode = args.split.split('_')[1] if len(args.split.split('_'))>1 else ''
        prefix = f"{mode}-" if len(mode) > 0 else ""
        save_gpt_neo_path = os.path.join(args.save_gpt_neo_prefix, prefix+f"{traindev}.jsonl")
        write_jsonl(gpt_neo_item, save_gpt_neo_path)
        print("number of output data for gpt neo: {}\n".format(len(gpt_neo_item)))

