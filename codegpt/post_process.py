import argparse
import bleu
import json
import re

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # '/home/t-jhuang/home/save/test/new_checkpoint/result/split_generation_results.json'
    parser.add_argument("--gold_file",
                        default='./result',
                        type=str, required=False, help="Path to save the generation results.")
    parser.add_argument("--generation_file",
                        default='./result',
                        type=str, required=False, help="Path to save the generation results.")
    parser.add_argument("--to_file",
                        default='./result',
                        type=str, required=False, help="Path to save the output.")
    args = parser.parse_args()

    with open(args.gold_file, 'r', encoding='utf-8') as fh:
        gold_file = fh.read().split('##########\n')
    with open(args.generation_file, 'r', encoding='utf-8') as fh:
        generation_file = fh.read().split('##########\n')
    if gold_file[-1] == '' and generation_file[-1] == '':
        gold_file = gold_file[:-1]
        generation_file = generation_file[:-1]
    print("reference_text: {}".format(len(gold_file)))
    print("translation_text: {}".format(len(generation_file)))
    outputs = []
    for target, generation in zip(gold_file, generation_file):
        item = {}
        item["input"] = ""
        item["target"] = target
        item["generation"] = generation
        outputs.append(item)
    print("output: {}".format(len(outputs)))
    write_json(outputs, args.to_file)


