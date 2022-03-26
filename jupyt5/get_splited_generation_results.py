import re
import json
import argparse

def write_json(file,path):
    with open(path,'w') as f:
        json.dump(file,f)

def takeFirst(elem):
    return elem[0]


parser = argparse.ArgumentParser()
parser.add_argument("--fairseq_generate_result_path", default='../saved_models/jupyt5/generate-test.txt', type=str, required=False, help="Path of dir to save the generation results.")
parser.add_argument("--write_path", default='../saved_models/jupyt5/split_generation_results.json', type=str, required=False, help="Path to save the splited generation results.")

args = parser.parse_args()

f = open(args.fairseq_generate_result_path)
line = f.readline()
S = []
T = []
H = []
while line:
    line = f.readline()
    if len(line) < 1:
        break
    if line[0] == 'S':
        number = int(line.split('\t')[0].split('-')[-1])
        sen = line.split('\t')[-1]
        S.append((number,sen))
    elif line[0] == 'T':
        number = int(line.split('\t')[0].split('-')[-1])
        sen = line.split('\t')[-1]
        T.append((number,sen))
    elif line[0] == 'H':
        number = int(line.split('\t')[0].split('-')[-1])
        sen = line.split('\t')[-1]
        H.append((number,sen))
f.close()

S = sorted(S,key=takeFirst)
T = sorted(T,key=takeFirst)
H = sorted(H,key=takeFirst)

final_list = []
for s,t,h in zip(S,T,H):
    item = {'input':s[1],'target':t[1],'generation':h[1]}
    final_list.append(item)
    
write_json(final_list,args.write_path)
