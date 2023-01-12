#! /usr/bin/env python

from ctypes import ArgumentError
import json
from pathlib import Path
from datetime import datetime
import os
from distutils.dir_util import copy_tree
import shutil
import signal
import sys
import argparse
from glob import glob
import logging
import subprocess

import yaml
from yaml.loader import SafeLoader


batch_script_version = "0.2.0"
program_name='moma_batch_run'

def print_batch_version_to_log():
    print(f'Batch script version: {batch_script_version}')

def query_yes_no(question, default="yes", trailing_string=""):
    "MM-20220809: This was taken from: https://stackoverflow.com/a/3041990"

    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = "[y/n] "
    elif default == "yes":
        prompt = "[Y/n] "
    elif default == "no":
        prompt = "[y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        getLogger().warning(question + prompt + trailing_string)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

"""
This class was taken from here: https://stackoverflow.com/a/39215961
"""
class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def add_gl_path(gl_ind, gl_entry, pos_ind, pos_entry, config):
    input_path = config['preprocessing_path']
    gl_path=input_path
    gl_path+=("/Pos"+str(pos_ind))
    gl_path+="/Pos"+str(pos_ind)+"_"+"GL"+str(gl_ind)
    gl_entry.update({'gl_path': gl_path})
    return gl_entry

"""
This method caluclates the paths to the GL directories.
"""
def get_gl_paths_for_position(input_path, positions, gl_paths, pos, config):
    if positions[pos]: # GLs are defined for this position; iterate over them to generate list of paths
        for gl in positions[pos]['gl']:
            gl_path=input_path
            gl_path+=("/Pos"+str(pos))
            gl_path+="/Pos"+str(pos)+"_"+"GL"+str(gl)
            gl_paths.append(gl_path)
            if not config['pos'][pos]['gl'][gl]:
                config['pos'][pos]['gl'][gl] = {}
            config['pos'][pos]['gl'][gl].update({'gl_path': gl_path})

def for_each_gl_in_config(config: dict, fnc):
    positions = config['pos']
    for pos_ind in positions:
        if positions[pos_ind]: # GLs are defined for this position; iterate over them to generate list of paths
            for gl_ind in positions[pos_ind]['gl']:
                updated_gl_entry = fnc(gl_ind, positions[pos_ind]['gl'][gl_ind], pos_ind, positions[pos_ind], config)
                if updated_gl_entry is not None: # only overwrite gl_entry, if method returns an updated version; this is not the case for e.g. validation methods
                    config['pos'][pos_ind]['gl'][gl_ind] = updated_gl_entry

def add_pos_and_gl_ind(gl_ind, gl_entry, pos_ind, pos_entry, config):
    gl_entry['gl_ind'] = gl_ind
    gl_entry['pos_ind'] = pos_ind
    return gl_entry

def build_arg_string(arg_dict):
    return ' '.join([f'-{key} {arg_dict[key]}' if arg_dict[key] is not None or '' else f'-{key}' for key in arg_dict])

class AnalysisMetadata(object):
    def __init__(self, path: Path):
        # assert type(path) is Path, f'path is not of type Path'
        self.__path = path
        if path.exists():
            with open(path, 'r') as fp:
                self.value_dict = json.load(fp)
        else:
            self.value_dict = {'file_version': '0.1.0',
            'created': datetime.now(),
            'tracked': False,
            'curated': False}
            # self.save()
        
    @property
    def path(self):
        return self.__path
    
    @property
    def tracked(self):
        return self.value_dict['tracked']

    @tracked.setter
    def tracked(self, val):
        self.value_dict['tracked'] = val
        self.save()

    @property
    def curated(self):
        return self.value_dict['curated']

    @curated.setter
    def curated(self, val):
        self.value_dict['curated'] = val
        self.save()

    def save(self):
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as fp:
            json.dump(self.value_dict, fp, indent=2, default=str)  # default=str is needed for the serialization of datetime object

class GlFileManager(object):
    def __init__(self, gl_directory_path: str, analysisName: str):
        self.gl_directory_path = Path(gl_directory_path)
        self.analysisName = analysisName

    def copy_track_data_to_backup_if_it_exists(self, backup_dir_postfix):
        if self.get_gl_track_data_path().exists():
            self.copy_track_data_to_backup(backup_dir_postfix)

    def copy_track_data_to_backup(self, backup_dir_postfix):
        self.copy_to_backup(self.get_gl_track_data_path(), backup_dir_postfix)

    def copy_to_backup(self, path_to_backup: Path, backup_dir_postfix: str):
        if path_to_backup.exists():
            backup_path = Path(str(path_to_backup) + backup_dir_postfix)
            copy_tree(str(path_to_backup), str(backup_path))

    def move_track_data_to_backup_if_it_exists(self, backup_dir_postfix):
        if self.get_gl_track_data_path().exists():
            self.move_track_data_to_backup(backup_dir_postfix)

    def move_track_data_to_backup(self, backup_dir_postfix):
        self.move_to_backup(self.get_gl_track_data_path(), backup_dir_postfix)

    def move_export_data_to_backup_if_it_exists(self, backup_dir_postfix):
        if self.get_gl_export_data_path().exists():
            self.move_export_data_to_backup(backup_dir_postfix)

    def move_export_data_to_backup(self, backup_dir_postfix: str):
        self.move_to_backup(self.get_gl_export_data_path(), backup_dir_postfix)

    def move_to_backup(self, path_to_backup: Path, backup_dir_postfix: str):
        if path_to_backup.exists():
            backup_path = Path(str(path_to_backup) + backup_dir_postfix)
            os.rename(path_to_backup, backup_path)

    def get_tiff_path(self) -> Path:
        gl_path = self.get_gl_directory_path()
        return glob(str(gl_path)+'/*[0-9].tif')[0]

    def get_gl_directory_path(self) -> Path:
        return self.gl_directory_path

    def get_gl_analysis_path(self) -> Path:
        path = Path(os.path.join(self.get_gl_directory_path(), self.analysisName))
        return path

    def get_gl_export_data_path(self) -> Path:
        return Path(os.path.join(self.get_gl_analysis_path(), 'export_data__' + self.analysisName))

    def get_gl_track_data_path(self) -> Path:
        return Path(os.path.join(self.get_gl_analysis_path(), 'track_data__' + self.analysisName))

    def get_gl_analysis_log_file_path(self) -> Path:
        return self.get_gl_track_data_path().joinpath('moma.log')

    def get_gl_is_curated(self) -> bool:
        return self.__get_analysis_metadata().curated

    def get_gl_is_tracked(self):
        return self.__get_analysis_metadata().tracked
    
    def get_gl_is_exported(self) -> bool:
        return self.get_gl_export_data_path().exists()
    
    def set_gl_is_tracked(self):
        self.__get_analysis_metadata().tracked = True
    
    def __get_analysis_metadata(self) -> AnalysisMetadata:
        return AnalysisMetadata(self.get_analysis_meta_data_path())

    def get_analysis_meta_data_path(self) -> Path:
        return Path(os.path.join(self.get_gl_track_data_path(), 'analysis_metadata.json'))

    def get_analysis_name(self):
        return self.analysisName

    def set_gl_is_curated(self):
        self.__get_analysis_metadata().tracked = True  # TODO-MM-20220808: this will be the case, if we were able to curate the GL; I add this here to handle GLs that we tracked, before implementing the use of `analysis_metadata.json`
        self.__get_analysis_metadata().curated = True

def get_list_of_default_args(config, list_of_gl_paths):
    if 'default_moma_arg' not in config:
        getLogger().error("'default_moma_arg' is not defined in 'yaml_config_file'")
        sys.exit(-1)

    cmd_args_dict_list = [{}]*len(list_of_gl_paths)
    for arg_dict in cmd_args_dict_list:
        arg_dict.update(config['default_moma_arg'])
    return cmd_args_dict_list

def initialize_gl_entry_to_dict(gl_ind, gl_entry, pos_ind, pos_entry, config):
    if gl_entry is None:
        return {}
    else:
        return gl_entry

def all_default_args_were_overwritten(gl_moma_arg, default_moma_arg):
    for arg in default_moma_arg:
        if arg not in gl_moma_arg and not arg == 'analysis':
            return False
    return True

def validate_moma_arg(gl_moma_arg, default_moma_arg):
    if 'analysis' in gl_moma_arg:
        raise ArgumentError("Nested instance of 'moma_arg' is not allowed to overwrite 'analysis' argument.")
    if not all_default_args_were_overwritten(gl_moma_arg, default_moma_arg):
        raise ArgumentError("Nested instance of 'moma_arg' must overwrite all values in 'default_moma_arg' (except for the 'analysis' value).")

def validate_moma_args(gl_ind, gl_entry, pos_ind, pos_entry, config):
    if 'moma_arg' in pos_entry:
        try:
            validate_moma_arg(pos_entry['moma_arg'], config['default_moma_arg'])
        except ArgumentError as e:
            getLogger().error(f'YAML config error in Pos {{{pos_ind}}}: ' + str(e))
            sys.exit(-1)
    if 'moma_arg' in gl_entry:
        try:
            validate_moma_arg(gl_entry['moma_arg'], config['default_moma_arg'])
        except ArgumentError as e:
            getLogger().error(f'YAML config error in GL {{{pos_ind}:{gl_ind}}}: ' + str(e))
            sys.exit(-1)

def append_gl_dicts_with_gl_file_manager(gl_entry: dict, gl_dicts: list) -> list:
    gl_file_manager = GlFileManager(gl_entry['gl_path'], gl_entry['moma_arg']['analysis'])
    gl_entry['gl_file_manager'] = gl_file_manager
    gl_dicts.append(gl_entry)

def add_moma_args(gl_ind, gl_entry, pos_ind, pos_entry, config):
    if 'moma_arg' in gl_entry:
        pass # there is nothing to do here, because moma_args was set at the GL level, so we do not need to add it
    elif 'moma_arg' in pos_entry:
        gl_entry['moma_arg'] = pos_entry['moma_arg']
    else:
        gl_entry['moma_arg'] = config['default_moma_arg']
    gl_entry['moma_arg'].update({'analysis': config['default_moma_arg']['analysis']})  # always set the analysis name to the default name
    return gl_entry

def calculate_log_file_path(yaml_config_file_path: Path, batch_operation_type: str):
    log_folder = Path(os.path.join(yaml_config_file_path.stem + '_logs'))
    if not log_folder.exists():
        os.makedirs(log_folder)
    return Path(os.path.join(log_folder, yaml_config_file_path.parent,yaml_config_file_path.stem + '_' + batch_operation_type.lower() + '.log'))

def getLogger() -> logging.Logger:
    return logging.getLogger('default')

def parse_gl_selection_string(selection_string: str) -> dict:
    try:
        selection_dict = eval(selection_string)
    except SyntaxError:
        getLogger().error(f"Could not parse value for option '--select': {selection_string}")
        sys.exit(-1)
    return selection_dict

def keep_user_selected_gls(config: dict, selection: dict) -> dict:
    cfg = config
    selected_pos_ind = [key for key in selection]
    for pos_ind in selected_pos_ind:
        if pos_ind not in cfg['pos']:
            getLogger().error(f"Position index {pos_ind} not defined in 'yaml_config_file'")
            sys.exit(-1)
    cfg['pos'] = {pos_ind:cfg['pos'][pos_ind] for pos_ind in selected_pos_ind}
    for pos_ind in cfg['pos']:
        selected_gl_ind = [key for key in selection[pos_ind]]
        for gl_ind in selected_gl_ind:
            if gl_ind not in cfg['pos'][pos_ind]['gl']:
                getLogger().error(f"GL index {{{pos_ind}:{gl_ind}}} not defined in 'yaml_config_file'")
                sys.exit(-1)
        cfg['pos'][pos_ind]['gl'] = {gl_ind:cfg['pos'][pos_ind]['gl'][gl_ind] for gl_ind in selected_gl_ind}
    return cfg

console_stdout = sys.stdout
console_stderr = sys.stderr

def initialize_logger(log_file):
    ### Initialize and configure logging ###
    # instructions how to setup the logger to write to terminal can be found here:
    # https://docs.python.org/3.8/howto/logging-cookbook.html
    # and
    # https://stackoverflow.com/a/38394903
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('default').addHandler(console)
    logger = logging.getLogger('default')
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)


class MomaRunner(object):
    _moma_process: subprocess.Popen = None
    _return_code: int = 0
    
    def __init__(self):
        pass

    def abort(self):
        if self.is_running:
            getLogger().info("STOPPING MOMA.")
            self._moma_process.terminate()
            self._moma_process.kill()
            # self._moma_process.send_signal(signal.SIGINT)
            # self._moma_process.send_signal(signal.SIGKILL)

    def run(self, logger, gl_file_manager: GlFileManager, current_args_dict):
        args_string = build_arg_string(current_args_dict)
        args_string += f' -i {gl_file_manager.get_tiff_path()}'
        moma_command = f'moma {args_string}'
        logger.info("RUN MOMA: " + moma_command)
        
        log_path = gl_file_manager.get_gl_analysis_log_file_path()
        logger.info("LOG MOMA: " + str(log_path))

        # old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self._moma_process = subprocess.Popen(['moma'] + args_string.split(),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
        # signal.signal(signal.SIGINT, old_handler)
        for line in self._moma_process.stdout:
            # sys.stdout.write(line)
            console_stdout.write(line)
        self._moma_process.wait()
        self._return_code = self._moma_process.returncode
        logger.info("FINISHED MOMA.")

    @property
    def is_running(self) -> bool:
        if not self._moma_process:
            return False
        poll = self._moma_process.poll()
        if poll is None:
            return True
        else:
            return False

    @property
    def return_code(self):
        return self._return_code

def killSignalHandler(signum, frame, abortObject, moma_runner: MomaRunner):
    abortObject.abortSignaled = True
    getLogger().info("Ctrl-c was pressed. Stopping execution.")
    getLogger().info("USER REQUESTED ABORT. STOPPING EXECUTION.")
    
    # user_response = input()
    # if user_response is 'y':
    #     getLogger().info("User selected 'y'. Stopping execution.")
    #     getLogger().info("USER REQUESTED ABORT. STOPPING EXECUTION.")
    #     abortObject.abortSignaled = True
    #     moma_runner.abort()
    # else:
    #     getLogger().info("User selected 'n'. Continuing execution.")

class AbortObject(object):
    abortSignaled = False

def parse_gls_to_process(yaml_config_file, gl_user_selection):
    '''
    Parses the GLs that will be processed.
    '''
    with open(yaml_config_file) as f:
        config = yaml.load(f, Loader=SafeLoader)

    if gl_user_selection:
        config = keep_user_selected_gls(config, gl_user_selection)

    for_each_gl_in_config(config, initialize_gl_entry_to_dict)
    for_each_gl_in_config(config, validate_moma_args)
    for_each_gl_in_config(config, add_moma_args)
    for_each_gl_in_config(config, add_gl_path)
    for_each_gl_in_config(config, add_pos_and_gl_ind)
    gl_dicts = []
    for_each_gl_in_config(config, lambda gl_ind, gl_entry, pos_ind, pos_entry, config: append_gl_dicts_with_gl_file_manager(gl_entry, gl_dicts))
    return gl_dicts

def parse_cmd_arguments():
    ### parse command line arguments ###
    parser = argparse.ArgumentParser(prog=program_name)
    group = parser.add_argument_group('required (mutually exclusive) arguments')
    mxgroup_metavar_name="yaml_config_file"
    
    args_with_yaml_config_file_path = ['delete', 'track', 'curate', 'export']

    mxgroup = group.add_mutually_exclusive_group(required=True)
    mxgroup.add_argument('-help', action='help')
    mxgroup.add_argument('-version', action='version', version=f'%(prog)s {batch_script_version}')
    mxgroup.add_argument("-delete_gl_analysis", dest='delete', metavar=mxgroup_metavar_name,
                    help="delete analysis files of specified GLs; WARNING: this will remove ALL analysis-files for the GLs")
    mxgroup.add_argument("-track", metavar=mxgroup_metavar_name,
                    help="run batch-tracking of GLs")
    mxgroup.add_argument("-curate", metavar=mxgroup_metavar_name,
                    help="run interactive curation of GLs")
    mxgroup.add_argument("-export", metavar=mxgroup_metavar_name,
                    help="run batch-export of tracking results")
    parser.add_argument("-l", "--log", type=str,
                    help="path to the log-file for this batch-run; derived from 'yaml_config_file' and stored next to it, if not specified")
    parser.add_argument("-select", "--select", type=str,
                    help="run on selection of GLs specified in Python dictionary-format; GLs must be defined in 'yaml_config_file'; example: \"{0:{1,2}, 3:{4,5}}\", where 0, 3 are position indices and 1, 2, 4, 5 are GL indices")
    parser.add_argument("-f", "--force", action='store_true',
                    help="force the operation")
    parser.add_argument("-ff", "--fforce", action='store_true',
                    help="force operation when deleting data; e.g. with option '-delete-analysis'")
    cmd_args = parser.parse_args()

    args_dict = vars(cmd_args)
    for arg_name in args_with_yaml_config_file_path:
        if args_dict[arg_name]:
            cmd_args.yaml_config_file = Path(args_dict[arg_name]) # get YAML config file path from the arguments it as value
    return cmd_args

def __main__():
    cmd_args = parse_cmd_arguments()

    moma_runner = MomaRunner()

    abortObj = AbortObject()
    abortObj.abortSignaled = False
    signal.signal(signal.SIGINT, lambda signum, frame: killSignalHandler(signum, frame, abortObj, moma_runner))
    
    ### Get time stamp of current run; used e.g. in the name of backup files ###
    time_stamp_of_run = datetime.now().strftime('%Y%m%d-%H%M%S')

    yaml_config_file_path = Path(cmd_args.yaml_config_file)
    
    if not yaml_config_file_path.exists():
        getLogger().error("Check argument 'yaml_config_file'; file not found at: {yaml_config_file_path}")
        exit(-1)

    batch_operation_type = 'DELETE' if cmd_args.delete else 'TRACK' if cmd_args.track else 'CURATE' if cmd_args.curate else 'EXPORT' if cmd_args.export else 'UNDEFINED ERROR'
    
    if cmd_args.log is not None:
        log_file = Path(cmd_args.log)
    else:
        log_file = calculate_log_file_path(yaml_config_file_path, batch_operation_type)

    with open(log_file, 'a') as f:
        if not f.writable():
            if cmd_args.log is not None:
                getLogger().error("Check argument '-log'; cannot write to the log-file at: {cmd_args.log}")
                sys.exit(-1)
            else:
                getLogger().error("Cannot write to the file log-file at: {cmd_args.log}")
                sys.exit(-1)

    initialize_logger(log_file)

    running_on_selection = cmd_args.select is not None
    gl_user_selection = {}
    if running_on_selection:
        if cmd_args.select is "":
            getLogger().error("Value is empty for option '--select'.")
            sys.exit(-1)
        gl_user_selection = parse_gl_selection_string(cmd_args.select)
    
    gl_dicts = parse_gls_to_process(cmd_args.yaml_config_file, gl_user_selection)
    gls_to_process_string = '\n'.join([f"Pos{gl['pos_ind']}_GL{gl['gl_ind']}" for gl in gl_dicts])

    if cmd_args.force:
        reply = query_yes_no("Forced run will OVERWRITE existing data (option '-f/--force'). Do you want to continue?", "no")
        if not reply:
            getLogger().info("Aborting forced run, because user replied 'no'. ")
            sys.exit(-1)
        getLogger().info("Performing forced run.")

    if cmd_args.delete and cmd_args.fforce:
        reply = query_yes_no(f"You are about to DELETE the analysis-folders in the GLs listed below. Do you REALLY want to continue? ", "no", trailing_string=f"\nSelected GLs:\n{gls_to_process_string}")
        if not reply:
            getLogger().info("Aborting deletion run, because user replied 'no'. ")
            sys.exit(-1)
        getLogger().info("Performing forced run.")
    
    if cmd_args.delete and not cmd_args.fforce:
        getLogger().info("ERROR: Option '-delete-analysis' must be combined with option '-fforce'.")
        sys.exit(-1)

    getLogger().info("BATCH RUN STARTED.")
    print_batch_version_to_log()  # print version to the log-file for later reference
    getLogger().info(f"Run type: {batch_operation_type}")
    batch_command_string = ' '.join(sys.argv)
    getLogger().info(f"Command: {batch_command_string}")
    backup_postfix = "__BKP_" + time_stamp_of_run
    getLogger().info(f"Any backups created during this run are appended with postfix: {backup_postfix}")
    
    for gl in gl_dicts:
        gl_directory_path = gl['gl_path']
        gl_file_manager = gl['gl_file_manager']

        args_dict = gl['moma_arg']
        current_args_dict = args_dict.copy()
        
        if cmd_args.track:
            if not gl_file_manager.get_gl_is_tracked() or cmd_args.force:
                gl_file_manager.move_track_data_to_backup_if_it_exists(backup_postfix)
                gl_file_manager.move_export_data_to_backup_if_it_exists(backup_postfix)
                current_args_dict.update({'headless':None, 'trackonly':None})
                moma_runner.run(getLogger(), gl_file_manager, current_args_dict)
                gl_file_manager.set_gl_is_tracked()
            else:
                getLogger().warning(f"Will not perform operation {batch_operation_type} for this GL, because it was already tracked for analysis '{gl_file_manager.get_analysis_name()}' in directory: {gl_file_manager.get_gl_track_data_path()}")
        elif cmd_args.curate:
            if not gl_file_manager.get_gl_is_curated() or cmd_args.force:
                gl_file_manager.copy_track_data_to_backup_if_it_exists(backup_postfix)
                gl_file_manager.move_export_data_to_backup_if_it_exists(backup_postfix)
                current_args_dict = {'reload': gl_directory_path, 'analysis': gl_file_manager.get_analysis_name()}  # for running the curation we only need the GL directory path and the name of the analysis
                moma_runner.run(getLogger(), gl_file_manager, current_args_dict)
                gl_file_manager.set_gl_is_curated()
            else:
                getLogger().warning(f"Will not perform operation {batch_operation_type} for this GL, because it was already curated for this analysis '{gl_file_manager.get_analysis_name()}' in directory: {gl_file_manager.get_gl_export_data_path()}")
        elif cmd_args.export:
            if not gl_file_manager.get_gl_is_exported() or cmd_args.force:
                gl_file_manager.copy_track_data_to_backup_if_it_exists(backup_postfix)
                gl_file_manager.move_export_data_to_backup_if_it_exists(backup_postfix)
                current_args_dict = {'headless':None, 'reload': gl_directory_path, 'analysis': gl_file_manager.get_analysis_name()}  # for running the curation we only need the GL directory path and the name of the analysis
                moma_runner.run(getLogger(), gl_file_manager, current_args_dict)
            else:
                getLogger().warning(f"Will not perform operation {batch_operation_type} for this GL, because it was already exported for this analysis '{gl_file_manager.get_analysis_name()}' in directory: {gl_file_manager.get_gl_export_data_path()}")
        elif cmd_args.delete and cmd_args.fforce:
            if gl_file_manager.get_gl_analysis_path().exists():
                getLogger().info(f"User selected operation {batch_operation_type}: Deleting analysis '{gl_file_manager.get_analysis_name()}' from GL: {gl_file_manager.get_gl_directory_path()}")
                shutil.rmtree(gl_file_manager.get_gl_analysis_path())
        if abortObj.abortSignaled:
                break
    if abortObj.abortSignaled:
        getLogger().info("BATCH RUN ABORTED.")
    else:
        getLogger().info("BATCH RUN FINISHED.")

if __name__ == "__main__":
    __main__()