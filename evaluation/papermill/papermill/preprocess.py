from __future__ import unicode_literals, print_function
from future.utils import raise_from  # noqa: F401

import sys
import os
import re 
import urllib.request
import traceback
import signal
import shutil
from urllib.parse import quote

from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
from nbformat.v4 import output_from_msg

class ExecutionParam: 
    def __init__(self, error_log, execution_timeout, repo_path, continue_with_error, progress_log, missing_file_log, processor_id, keyword):
        self.error_log = error_log
        self.execution_timeout = execution_timeout
        self.repo_path = repo_path
        self.continue_with_error = continue_with_error
        self.progress_log = progress_log
        self.missing_file_log = missing_file_log
        self.processor_id = processor_id
        self.keyword = keyword

try:
    from queue import Empty  # Py 3
except ImportError:
    from Queue import Empty  # Py 2

def find_urls(string): 
    # findall() has been used  
    # with valid conditions for urls in string 
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string) 
    return url

class Timeout(Exception):
    pass
def MyHandler(sig, frame):
    raise Timeout

class CustomModuleNotFoundError(Exception):
    def __init__(self, message, module_name):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.module_name = module_name
    def try_fix(self, kernelname):
        #if kernelname == 'python3':
        #    os.system('python3 -m pip install {}'.format(self.module_name))
        #    os.system('pip3 install {}'.format(self.module_name))
        #else:
        #    os.system('python2 -m pip install {}'.format(self.module_name))
        #    os.system('pip2 install {}'.format(self.module_name))
        pass

class CustomNameNotFoundError(Exception):
    def __init__(self, message, name):
        self.message = message
        self.name = name
    def try_fix(self, cell):
        name = self.name
        if name in ['random','rand','rd']:
            cell.source = 'import random as {}\n'.format(name) + cell.source
            return True
        if name in ['pandas','pd']:
            cell.source = 'import pandas as {}\n'.format(name) + cell.source
            return True
        if name in ['sys', 'os']:
            cell.source = 'import {}\n'.format(name) + cell.source
            return True
        if name in ['plt']:
            cell.source = 'import matplotlib.pyplot as plt\n' + cell.source
            return True
        if name in ['np', 'numpy']:
            cell.source = 'import numpy as {}\n'.format(name) + cell.source
            return True
        return False

recognized_names = ['random','rand','rd','pandas','pd','sys','os','np','numpy','plt']     

class CustomFileNotFoundError(Exception):
    def __init__(self, keyword, message, file_name):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.file_name = file_name
        self.dataset = keyword
    def try_fix(self, cell, source_text, input_path, repo_path):
        filename = os.path.split(self.file_name)[1] # without path  
        filename = filename.split('\\')[-1]
        dataset_folder = '../datafiles/' #+ self.dataset #+'.'.join(filename.split('.')[:-1])
        dataset_folder_absolute = '/datadrive2/notebooks/datafiles/ ' #.format(self.dataset)
        #directory = os.path.basename(self.file_name)
        if cell.source.find(self.file_name) != -1:
            cell.source = cell.source.replace(self.file_name, str(os.path.join(os.getcwd(),dataset_folder,filename)))
        else:
            pattern = 'os.path.join\((.*)?{}(.*)?'.format(filename)
            pathj = re.search(pattern, cell.source)
            if pathj:
                cell.source = cell.source.replace(cell.source[pathj.start():pathj.end()-1], '"{}"'.format(str(os.path.join(os.getcwd(),dataset_folder,filename))))
        urls = find_urls(source_text)
        file_found = False
        new_file_locate = os.path.join(os.getcwd(),dataset_folder) #os.path.split(input_path)[0] 
        target_file = os.path.join(new_file_locate, filename)
        print("trying to find data file: {}, search from repo {}, target location = {}".format(self.file_name, repo_path, target_file))
        ## trial 1: find the matching files in the repo
        for root, dirs, files in os.walk(repo_path):
            for fn in files:
                if fn == filename and (not os.path.normpath(input_path)==os.path.normpath(os.path.join(root, filename))):
                    #os.rename(os.path.join(root, filename), target_file)
                    try:
                        shutil.copy(os.path.join(root, filename), target_file)
                        print('mv {} {}'.format(os.path.join(root, filename), target_file))
                        return True
                    except:
                        return False
        # trial 2.1: find under ../datafiles/dataset directory, if file exists, then no need to search
        if os.path.exists(target_file):
            return True
        ## trial 2.2: find under the jupyter notebook's working directory
        for fn in os.listdir('/datadrive2/notebooks/replay/'):
            if fn == filename:
                try:
                    shutil.copy(os.path.join('/datadrive2/notebooks/replay/',fn), target_file)
                    print("mv {} {}".format(os.path.join('/datadrive2/notebooks/replay/',fn), target_file))
                    return True
                except:
                    pass
        ## trial 2.3: find under the jupyter notebook's working directory
        #for fn in os.listdir('/home/azureuser/'):
        #    if fn == filename:
        #        try:
        #            shutil.copy(os.path.join('/home/azureuser/',fn), target_file)
        #            print("mv {} {}".format(os.path.join('/home/azureuser/',fn), target_file))
        #            return True
        #        except:
        #            pass
        # trial 3: find the matching files in the url in cells
        if not file_found:
            for url in urls:
                if filename in url:
                    i = url.find(filename)
                    if i != -1:
                      cleaned_url = url[:i+len(filename)]
                      try:
                        print("clone from {}".format(cleaned_url))
                        urllib.request.urlretrieve(url, target_file) 
                        file_found = True
                      except:
                        continue
        # trial 4: search on kaggle dataset
        if file_found == False:
            cmd =  'kaggle datasets list -s "{}"'.format(filename)
            r = os.popen(cmd).read()
            if not r.startswith('404'):
                repos = []
                start_repos = False
                for line in r.split('\n'):
                    if line.startswith('----'):
                        start_repos = True
                    elif start_repos:
                        x = list(filter(lambda t:t!='', line.split(' ')))
                        if len(x) > 0:
                            repos.append(x[0])
                if len(repos) > 0:
                    kaggle_repo = repos[0]
                    for krepo in repos:
                        repo_split = krepo.split('/')[-1]
                        if source_text.find(repo_split) != -1:
                            kaggle_repo = krepo 
                    print("download from kaggles: {} {}".format(kaggle_repo, filename))
                    download_cmd = 'kaggle datasets download {} --unzip --force --path "{}" -f "{}"'.format(kaggle_repo, dataset_folder_absolute, filename)
                    os.system(download_cmd)
                    downloaded_filename = quote(filename)
                    #if os.path.exists('{}.zip'.format(downloaded_filename)):
                    #    os.system("unzip {}.zip".format(downloaded_filename))
                    #    os.system('chmod 666 {}'.format(downloaded_filename))
                    os.rename(os.path.join(os.getcwd(), downloaded_filename), target_file)
                    file_found = True
        return file_found
               
        

class CustomCellExecutionError(Exception):
  def __init__(self, exception, recognized_error):
    self.exception = exception
    self.recognized_error = recognized_error

class PapermillExecutePreprocessor(ExecutePreprocessor):
    """Module containing a preprocessor that executes the code cells
    and updates outputs"""

    def preprocess(self, nb_man, resources, km=None):
        """
        Wraps the parent class process call slightly
        """
        with self.setup_preprocessor(nb_man.nb, resources, km=km):
            if self.log_output:
                self.log.info("Executing notebook with kernel: %s" % self.kernel_name)
            nb, resources = self.papermill_process(nb_man, resources)
            info_msg = self._wait_for_reply(self.kc.kernel_info())
            nb.metadata['language_info'] = info_msg['content']['language_info']

        return nb, resources

    def start_new_kernel(self, **kwargs):
        """Creates a new kernel manager and kernel client.
        Parameters
        ----------
        kwargs :
            Any options for `self.kernel_manager_class.start_kernel()`. Because
            that defaults to KernelManager, this will likely include options
            accepted by `KernelManager.start_kernel()``, which includes `cwd`.
        Returns
        -------
        km : KernelManager
            A kernel manager as created by self.kernel_manager_class.
        kc : KernelClient
            Kernel client as created by the kernel manager `km`.
        """
        if not self.kernel_name:
            self.kernel_name = self.nb.metadata.get(
                'kernelspec', {}).get('name', 'python')
        # TO REPLACE
        km = self.kernel_manager_class(kernel_name=self.kernel_name,
                                       config=self.config)
        km.start_kernel(extra_arguments=self.extra_arguments, **kwargs)

        # the rest are the same
        kc = km.client()
        kc.start_channels()
        try:
            kc.wait_for_ready(timeout=self.startup_timeout)
        except RuntimeError:
            kc.stop_channels()
            km.shutdown_kernel()
            raise
        kc.allow_stdin = False
        return km, kc
        km, kc = super(PapermillExecutePreprocessor, self).start_new_kernel(**kwargs)
        # Note sure if we need this anymore?
        kc.allow_stdin = False

        return km, kc

    def papermill_process(self, nb_man, resources):
        """
        This function acts as a replacement for the grandparent's `preprocess`
        method.

        We are doing this for the following reasons:

        1. Notebooks will stop executing when they encounter a failure but not
           raise a `CellException`. This allows us to save the notebook with the
           traceback even though a `CellExecutionError` was encountered.

        2. We want to write the notebook as cells are executed. We inject our
           logic for that here.

        3. We want to include timing and execution status information with the
           metadata of each cell.

        Parameters
        ----------
        nb_man : NotebookExecutionManager
            Engine wrapper of notebook being converted
        resources : dictionary
            Additional resources used in the conversion process.  Allows
            preprocessors to pass variables into the Jinja engine.

        """
        # Execute each cell and update the output in real time.
        nb = nb_man.nb
        file_source = ''
        cell_with_keyword = []
        #for index, cell in enumerate(nb.cells):
        #    file_source += cell.source
        #    if self.execution_param.keyword in str(cell.source):
        #        cell_with_keyword.append(index)
        def check_os_command(source):
            for l in source.split('\n'):
                if l.startswith('!'):
                    return True
            return False
        for index, cell in enumerate(nb.cells):
            later_cell_text = '\n'.join([str(c1.source) for c1 in nb.cells[index+1:]])
            #keywords_ = ['agg(','aggregate(','groupby(','pivot(','merge(','drop_duplicates(','melt(','pivot_table(','replace(','transform(','apply(','map(','applymap(','isin(','crosstab(','concat(','dropna(','corr(','cov(','stack(','unstack(','map(','get_dummies(','isnull','to_numeric', ']=', '] =']
            #keywords_ = ['groupby(','pivot','merge(','join(','drop_duplicates(','drop(','melt(','pivot_table(','replace(','transform(','apply(','applymap(','isin(','crosstab(','concat(','dropna(','fillna(','stack(','unstack(','map(','get_dummies(','cumsum',']=','] =']
            #if not any([kw in later_cell_text for kw in keywords_]):
            #    break
            #if len(cell_with_keyword) > 0 and index > cell_with_keyword[-1]:
            #    return nb, resources 
            try:
                #msg = '\tprogress cell start: {} / {}\n'.format(index, len(nb.cells))
                #log_to_process(self.execution_param.progress_log, msg)
                nb_man.cell_start(cell)
                if not cell.source:
                    #msg = '\tprogress cell (non-source) finish: {} / {}\n'.format(index, len(nb.cells))
                    #log_to_process(self.execution_param.progress_log, msg)
                    continue
                if 'os.system(' in cell.source or 'shutdown' in cell.source or 'sudo ' in cell.source or check_os_command(cell.source):
                    #print('FIND COMMAND: {} {} {} {}'.format('os.system(' in cell.source, 'shutdown' in cell.source, 'sudo ' in cell.source, check_os_command(cell.source)))
                    #msg = '\tprogress cell including system command not executed: {} / {}\n'.format(index, len(nb.cells))
                    #log_to_process(self.execution_param.progress_log, msg)
                    continue
                nb.cells[index], resources = self.preprocess_cell(cell, resources, index, file_source, nb.metadata.kernelspec.name)
                #nb.cells[index], resources = self.preprocess_cell(cell, resources, index, file_source, nb.metadata.kernelspec.name)
                #msg = '\tprogress cell finish: {} / {}\n'.format(index, len(nb.cells))
                #log_to_process(self.execution_param.progress_log, msg)
            except TimeoutError as ex:
                nb_man.cell_exception(nb.cells[index], exception=ex)
                break
                #if self.execution_param.continue_with_error:
                #  continue
                #else:
                #  break
            except CellExecutionError as ex:
                nb_man.cell_exception(nb.cells[index], exception=ex)
                if self.execution_param.continue_with_error:
                  continue
                else:
                  break
            except CustomCellExecutionError as ex:
                nb_man.cell_exception(nb.cells[index], exception=ex.exception, recognized_error=ex.recognized_error)
                if self.execution_param.continue_with_error:
                  continue
                else:
                  break
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                if self.execution_param.continue_with_error:
                    continue
                else:
                    raise e
            finally:
                nb_man.cell_complete(nb.cells[index])
                orig_pickle_file = '/home/azureuser/temp/temp_{}.pickle'.format(self.execution_param.processor_id)
                if os.path.exists(orig_pickle_file):
                    try:
                        cellid = cell.execution_count if hasattr(cell, 'execution_count') and cell.execution_count is not None else index
                        shutil.copy(orig_pickle_file, '/datadrive2/notebooks/dumped_variables/{}_cell{}.pickle'.format(self.input_path.replace('/','_').replace('.ipynb',''), cellid))
                        fp1 = open('/datadrive2/notebooks/cell_source_code/{}_cell{}.py'.format(self.input_path.replace('/','_').replace('.ipynb',''), cellid),'w')
                        fp1.write(str(cell.source))
                        fp1.close()
                        os.system('rm {}'.format(orig_pickle_file))
                    except:
                        traceback.print_exc(file=sys.stdout)
                        os.system('rm {}'.format(orig_pickle_file))
                        pass
        return nb, resources

    def log_output_message(self, output):
        if output.output_type == "stream":
            if output.name == "stdout":
                self.log.info("".join(output.text))
            elif output.name == "stderr":
                # In case users want to redirect stderr differently, pipe to warning
                self.log.warning("".join(output.text))
        elif "data" in output and "text/plain" in output.data:
            self.log.info("".join(output.data['text/plain']))
        # Force a flush to avoid long python buffering for messages
        sys.stdout.flush()
        sys.stderr.flush()

    def preprocess_cell(self, cell, resources, cell_index, file_source='', kernelname='python3'):
        """
        Executes a single code cell. See base.py for details.
        To execute all cells see :meth:`preprocess`.
        """
        if cell.cell_type != 'code' or not cell.source.strip():
            return cell, resources
        def check_same_error(errors, e):
            if len(errors) == 0:
                return False
            if isinstance(errors[-1], CustomModuleNotFoundError) and isinstance(e, CustomModuleNotFoundError) and errors[-1].module_name == e.module_name:
                return True
            if isinstance(errors[-1], CustomFileNotFoundError) and isinstance(e, CustomFileNotFoundError) and errors[-1].file_name == e.file_name:
                return True
            if isinstance(errors[-1], CustomNameNotFoundError) and isinstance(e, CustomNameNotFoundError) and errors[-1].name == e.name:
                return True
            return False
            
        def log_error(e):
            if isinstance(e, CustomFileNotFoundError):
                self.execution_param.missing_file_log.write('missing {}\n'.format(e.file_name))
                self.execution_param.missing_file_log.flush()

        recognized_error = None
        raise_fixable_error = True
        try_count = 0
        errors = []
        while try_count < 10:
          try:
            signal.signal(signal.SIGALRM, MyHandler)
            signal.alarm(self.timeout)
            reply, outputs, recognized_error_tmp = self.run_cell(cell, cell_index, raise_fixable_error)
            #recognized_error = recognized_error_tmp if recognized_error_tmp else recognized_error
            signal.alarm(0)
            break
          except Timeout:
            print("cell {} takes too long".format(cell_index))
            signal.alarm(0)
            outputs = []
            reply = None
            raise TimeoutError()
            break
          except CustomModuleNotFoundError as e:
            if check_same_error(errors, e):
                raise_fixable_error = False
            else:
                errors.append(e)
                # install package here
                e.try_fix(kernelname)
                try_count += 1
          except CustomFileNotFoundError as e:
            if check_same_error(errors, e):
                log_error(e)
                raise_fixable_error = False
            else:
                errors.append(e)
                # try to copy file or download from online
                r = e.try_fix(cell, file_source, self.input_path, self.execution_param.repo_path)
                if r == False:
                    log_error(e)
                    raise_fixable_error = False
                try_count += 1
          except CustomNameNotFoundError as e:
            if check_same_error(errors, e):
                log_error(e)
                raise_fixable_error = False
            else:
                errors.append(e)
                r = e.try_fix(cell)
                if r == False:
                    raise_fixable_error = False
          finally:
            signal.alarm(0)
          
        signal.alarm(0)
        cell.outputs = outputs

        cell_allows_errors = (self.allow_errors or "raises-exception"
                              in cell.metadata.get("tags", []))

        if self.force_raise_errors or not cell_allows_errors:
            for out in outputs:
                if out.output_type == 'error':
                    er = CellExecutionError.from_cell_and_msg(cell, out)
                    raise er 
            if (reply is not None) and reply['content']['status'] == 'error':
                raise CellExecutionError.from_cell_and_msg(cell, reply['content'])
        return cell, resources

    # TODO: Update nbconvert to allow for msg yielding so we can log as messages arrive
    def run_cell(self, cell, cell_index=0, raise_fixable_error=False):

        global recognized_names
        msg_id = self.kc.execute(cell.source)
        # Log check added to original implementation
        if self.log_output:
            self.log.info('Executing Cell {:-<40}'.format(cell_index + 1))
        self.log.debug("Executing cell contents:\n%s", cell.source)
        outs = cell.outputs = []
        fixable_error = None
        recognized_error = None

        while True:
            try:
                # We are not waiting for execute_reply, so all output
                # will not be waiting for us. This may produce currently unknown issues.
                msg = self.kc.iopub_channel.get_msg(timeout=self.timeout-10) # None / self.iopub_timeout
            except Empty:
                self.log.warning("Timeout waiting for IOPub output")
                if self.raise_on_iopub_timeout:
                    raise RuntimeError("Timeout waiting for IOPub output")
                else:
                    break
            if msg['parent_header'].get('msg_id') != msg_id:
                # not an output from our execution
                continue

            msg_type = msg['msg_type']
            self.log.debug("output: %s", msg_type)
            content = msg['content']

            if 'ename' in content:
              if "No module named" in content['evalue']:
                missing_module = content['evalue'].replace('No module named ', '').replace("'",'').replace('"','')
                fixable_error = CustomModuleNotFoundError(content['evalue'], missing_module)
              
              elif content['ename'] == 'FileNotFoundError' or content['ename'] == 'IOError':
                missing_file = content['evalue'].split("'")[-2]
                fixable_error = CustomFileNotFoundError(self.execution_param.keyword, content['evalue'], missing_file)

              elif content['ename'] == 'NameError':
                name = content['evalue'].replace('name ','').replace('global','').replace(' is not defined','').replace('"','').replace("'",'').replace(' ','')
                if name in recognized_names:
                    fixable_error = CustomNameNotFoundError(content['evalue'],name)
                

            # set the prompt number for the input and the output
            if 'execution_count' in content:
                cell['execution_count'] = content['execution_count']

            if msg_type == 'status':
                if content['execution_state'] == 'idle':
                    break
                else:
                    continue
            elif msg_type == 'execute_input':
                continue
            elif msg_type == 'clear_output':
                outs[:] = []
                # clear display_id mapping for this cell
                for display_id, cell_map in self._display_id_map.items():
                    if cell_index in cell_map:
                        cell_map[cell_index] = []
                continue
            elif msg_type.startswith('comm'):
                continue

            display_id = None
            if msg_type in {'execute_result', 'display_data', 'update_display_data'}:
                display_id = msg['content'].get('transient', {}).get('display_id', None)
                if display_id:
                    self._update_display_id(display_id, msg)
                if msg_type == 'update_display_data':
                    # update_display_data doesn't get recorded
                    continue

            try:
                out = output_from_msg(msg)
            except ValueError:
                self.log.error("unhandled iopub msg: " + msg_type)
                continue
            if display_id:
                cell_map = self._display_id_map.setdefault(display_id, {})
                output_idx_list = cell_map.setdefault(cell_index, [])
                output_idx_list.append(len(outs))

            # Log check added to original implementation
            if self.log_output:
                self.log_output_message(out)
            outs.append(out)

        if raise_fixable_error and fixable_error:
          raise fixable_error

        exec_reply = self._wait_for_reply(msg_id, cell)
        # Log check added to original implementation
        if self.log_output:
            self.log.info('Ending Cell {:-<43}'.format(cell_index + 1))
            # Ensure our last cell messages are not buffered by python
            sys.stdout.flush()
            sys.stderr.flush()

        return exec_reply, outs, recognized_error

def log_to_process(progress_log_file, msg):
    if progress_log_file:
        progress_log_file.write(msg)
        progress_log_file.flush()
    print(msg)
