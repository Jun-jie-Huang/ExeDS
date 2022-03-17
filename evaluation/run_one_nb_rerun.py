import os
import sys
import traceback
sys.path.append('./papermill/')
import papermill as pm
import argparse
import json


def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


# process_id = int(sys.argv[4])
def process_args(args):
    args.error_log_filename = os.path.join(args.base_dir, './error_{}.log'.format(args.process_id))
    args.progress_log = open(os.path.join(args.base_dir, './progress_{}.log'.format(args.process_id)), 'a+')
    args.failed_log = open(os.path.join(args.base_dir, 'failed_files_{}.log'.format(args.process_id)), 'a+')
    args.missing_file_log = open(os.path.join(args.base_dir, 'missing_file_{}.log'.format(args.process_id)), 'a+')
    args.idx_path = os.path.join(args.base_dir, args.idx_path)
    return args


def try_run_one_notebook(repo_path, nb_name, nbid):
    exec_param = pm.ExecutionParam(args.error_log_filename, 1000, repo_path, True, args.progress_log,
                                   args.missing_file_log, args.process_id, nbid)
    fname = os.path.join(repo_path, nb_name)

    # fix kernel
    content = json.load(open(fname, 'r'))
    if content["metadata"]["kernelspec"]["name"] not in ['python2', 'python3']:
        orig_name = content["metadata"]["kernelspec"]["name"]
        if content["metadata"]['language_info']['version'] == 2:
            kernel = 'python2'
        else:
            kernel = 'python3'
        print("replace kernal from {} to {}".format(orig_name, kernel))
        nb_content = ''.join([line for line in open(fname, 'r')])
        nb_content = nb_content.replace('"name": "{}"'.format(orig_name), '"name": "{}"'.format(kernel))
        with open(fname, 'w') as fpw:
            fpw.write(nb_content)

    print("running {}".format(fname))
    output_notebook = os.path.join(repo_path, 'run_{}'.format(nb_name))
    try:
        rt = pm.execute_notebook(fname, output_notebook, execution_param=exec_param)
    except Exception as e:
        args.failed_log.write('FAIL {}: error = {}\n'.format(fname, str(e)))
        traceback.print_exc(file=args.failed_log)
        print("Error: {}".format(e))
        traceback.print_exc(file=sys.stdout)
        args.failed_log.flush()
    args.progress_log.flush()


# repo_path = sys.argv[1]
# nb_name = sys.argv[2]
# nbid = sys.argv[3]
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='./ExeDS_notebook/', type=str)
parser.add_argument('--idx_path', default='idx2dir.json', type=str)
parser.add_argument('--process_id', default=0, type=int)

args = parser.parse_args()
sys.path.append(args.base_dir)

args = process_args(args)
idx2dir = read_json(args.idx_path)

for instance in idx2dir:
    print("### Executing: no.{}/{},  \tdir:{}".format(instance['idx'], len(idx2dir), instance['dir']))
    repo_path = os.path.join(args.base_dir, instance['dir'])
    nbid = instance['dir'].split('_')[1]
    nb_name = "file_{}.ipynb".format(nbid)
    try_run_one_notebook(repo_path, nb_name, nbid)
# os.system('rm -rf ../datafiles/*')





