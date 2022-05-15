import argparse
import json
import re

def read_json(name):
    with open(name, 'r') as f:
        json_file = json.load(f)
    return json_file


def write_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f)


def post_process_gptneo_generaion(item):
    split_generation = re.split(r'(# Code of the context is: |# Context is: |# TABLE |# MARKDOWN |# TABLE |# CODE)', item['generation'])
    # # Use the code after '# Code of the context is: '
    # for idx, gene in enumerate(split_generation):
    #     if gene == '# Code of the context is: ':
    #         code = split_generation[idx+1]
    # # Use the code first
    if len(split_generation) < 2:
        print('len == 1')
    elif split_generation[1] != '# Context is: ':
        print("Mismatch!")
    code = split_generation[0]
    return code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_generation",
                        default="../../exejuice/saved_models/gpt_neo/recheck_addTab-df_madeup_ori_token_range3_lineLen1-25_c200m200a900_epoch10_block2048_bz3_G16_normal/split_generation_results.json",
                        type=str, help="The path to generation output .")
    parser.add_argument("--to_path",
                        default="../../exejuice/saved_models/gpt_neo/recheck_addTab-df_madeup_ori_token_range3_lineLen1-25_c200m200a900_epoch10_block2048_bz3_G16_normal/split_generation_results_post_process.json",
                        type=str, required=False, help="Path to save the output.")
    args = parser.parse_args()

    generations = read_json(args.path_generation)

    outputs = []
    for generation in generations:
        item = {}
        item["input"] = generation['input']
        item["target"] = generation['target']
        item["generation"] = post_process_gptneo_generaion(generation)
        outputs.append(item)
    print("output: {}".format(len(outputs)))
    write_json(outputs, args.to_path)


