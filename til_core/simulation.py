# encoding: utf-8
from __future__ import print_function, division


from collections import defaultdict
import datetime
import pkg_resources
import os.path
import random
import subprocess
import time
import warnings

import numpy as np
import tables
import yaml

# Monkey patching liam2
from liam2.exprtools import functions

from til_core import exprmisc
functions.update(exprmisc.functions)

# TilSimulation specific import
from liam2 import config, console
from liam2.context import EvaluationContext
from liam2.data import VoidSource, H5Source, H5Sink
from liam2.entities import Entity, global_symbols
from liam2.utils import (time2str, timed, gettime, validate_dict, field_str_to_type, fields_yaml_to_type,
     UserDeprecationWarning)
from liam2.simulation import (expand_periodic_fields, handle_imports, show_top_processes, Simulation)

# from til.utils import addmonth, time_period
# from til.process import ExtProcess

def get_git_head_revision(distribution_name):
    distribution_location = pkg_resources.get_distribution(distribution_name).location
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd = distribution_location)
    return git_hash.strip()

class TilSimulation(Simulation):
    index_for_person_variable_name_by_entity_name = {
        'individus': 'id',
        'foyers_fiscaux': 'idfoy',
        'menages': 'idmen',
        }  # TODO: should be imporved

    input_store = None
    output_store = None

    weight_column_name_by_entity_name = {
        'menages': 'wprm',
        }  # TODO should be elsewhere

    uniform_weight = None

    yaml_layout = {
        'import': None,
        'globals': {
            'periodic': None,  # either full-blown (dict) description or list
                               # of fields
            'weight': None,
            '*': {
                'path': str,
                'type': str,
                'fields': [{
                    '*': None  # Or(str, {'type': str, 'initialdata': bool, 'default': type})
                    }],
                'oldnames': {
                    '*': str
                    },
                'newnames': {
                    '*': str
                    },
                'invert': [str],
                'transposed': bool,
                },
            },
        '#entities': {
            '*': {
                'fields': [{
                    '*': None
                    }],
                'links': {
                    '*': {
                        '#type': str,  # Or('many2one', 'one2many', 'one2one')
                        '#target': str,
                        '#field': str
                        }
                    },
                'macros': {
                    '*': None
                    },
                'processes': {
                    '*': None
                    }
                }
            },
        '#simulation': {
            'init': [{
                '*': [str]
                }],
            'processes': [{
                '*': [None]  # Or(str, [str, int])
                }],
            'random_seed': int,
            'input': {
                'path': str,
                'file': str,
                'method': str
                },
            'output': {
                'path': str,
                'file': str
                },
            'legislation': {
                '#ex_post': bool,
                '#annee': int
                },
            'final_stat': bool,
#            'time_scale': str,
            'retro': bool,
            'logging': {
                'timings': bool,
                'level': str,  # Or('periods', 'functions', 'processes')
            },
            '#periods': int,
            'start_period': int,
            'init_period': int,
            'skip_shows': bool,
            'timings': bool,    # deprecated
            'assertions': str,  # Or('raise', 'warn', 'skip')
            'default_entity': str,
            'autodump': None,
            'autodiff': None,
            'runs': int,
            }
        }

    def __init__(self, globals_def, periods, start_period, init_processes,
                 processes, entities, input_method, input_path, output_path,
                 default_entity=None, runs=1, legislation = None, final_stat = False,
                 uniform_weight = None, config_log = None):
        if 'periodic' in globals_def:
            declared_fields = globals_def['periodic']['fields']
            fnames = {fname for fname, type_ in declared_fields}
            if 'PERIOD' not in fnames:
                declared_fields.insert(0, ('PERIOD', int))

        self.globals_def = globals_def
        self.periods = periods

        self.start_period = start_period
        # init_processes is a list of tuple: (process, 1)
        self.init_processes = init_processes
        # processes is a list of tuple: (process, periodicity, start)
        self.processes = processes
        self.entities = entities

        if input_method == 'h5':
            data_source = H5Source(input_path)
        elif input_method == 'void':
            data_source = VoidSource()
        else:
            raise ValueError("'%s' is an invalid value for 'method'. It should "
                             "be either 'h5' or 'void'")

        self.data_source = data_source
        self.data_sink = H5Sink(output_path)
        self.default_entity = default_entity

        self.stepbystep = False
        self.runs = runs

        self.uniform_weight = uniform_weight
        self.save_log(config_log)
        self.save_git_hash()

    @classmethod
    def from_str(cls, yaml_str, simulation_dir='',
                 input_dir=None, input_file=None,
                 output_dir=None, output_file=None,
                 start_period=None, periods=None, seed=None,
                 skip_shows=None, skip_timings=None, log_level=None,
                 assertions=None, autodump=None, autodiff=None,
                 runs=None, uniform_weight=None):
        content = yaml.load(yaml_str)
        expand_periodic_fields(content)
        content = handle_imports(content, simulation_dir)

        validate_dict(content, cls.yaml_layout)
        config_log = content.copy()
        # the goal is to get something like:
        # globals_def = {'periodic': {'fields': [('a': int), ...], ...},
        #                'MIG': {'type': int}}
        globals_def = {}
        for k, v in content.get('globals', {}).iteritems():
            if k == 'weight':
               pass
            elif "type" in v:
                v["type"] = field_str_to_type(v["type"], "array '%s'" % k)
            else:
                # TODO: fields should be optional (would use all the fields
                # provided in the file)
                v["fields"] = fields_yaml_to_type(v["fields"])
            globals_def[k] = v
        simulation_def = content['simulation']
        if seed is None:
            seed = simulation_def.get('random_seed')
        if seed is not None:
            seed = int(seed)
            print("using fixed random seed: %d" % seed)
            random.seed(seed)
            np.random.seed(seed)

        if periods is None:
            periods = simulation_def['periods']
        if start_period is None:
            start_period = simulation_def['start_period']

        if skip_shows is None:
            skip_shows = simulation_def.get('skip_shows', config.skip_shows)
        config.skip_shows = skip_shows
        if assertions is None:
            assertions = simulation_def.get('assertions', config.assertions)
        # TODO: check that the value is one of "raise", "skip", "warn"
        config.assertions = assertions

        logging_def = simulation_def.get('logging', {})
        if log_level is None:
            log_level = logging_def.get('level', config.log_level)
        config.log_level = log_level
        if config.log_level == 'procedures':
            config.log_level = 'functions'
            warnings.warn("'procedures' logging.level is deprecated, "
                          "please use 'functions' instead",
                          UserDeprecationWarning)

        if 'timings' in simulation_def:
            warnings.warn("simulation.timings is deprecated, please use "
                          "simulation.logging.timings instead",
                          UserDeprecationWarning)
            config.show_timings = simulation_def['timings']

        if skip_timings:
            show_timings = False
        else:
            show_timings = logging_def.get('timings', config.show_timings)
        config.show_timings = show_timings

        if autodump is None:
            autodump = simulation_def.get('autodump')
        if autodump is True:
            autodump = 'autodump.h5'
        if isinstance(autodump, basestring):
            # by default autodump will dump all rows
            autodump = (autodump, None)
        config.autodump = autodump

        if autodiff is None:
            autodiff = simulation_def.get('autodiff')
        if autodiff is True:
            autodiff = 'autodump.h5'
        if isinstance(autodiff, basestring):
            # by default autodiff will compare all rows
            autodiff = (autodiff, None)
        config.autodiff = autodiff

        legislation = simulation_def.get('legislation', None)
        final_stat = simulation_def.get('final_stat', None)

        input_def = simulation_def.get('input')
        if input_def is not None or input_dir is not None:
            input_directory = input_dir if input_dir is not None else input_def.get('path', '')
        else:
            input_directory = ''

        if not os.path.isabs(input_directory):
            input_directory = os.path.join(simulation_dir, input_directory)
        config.input_directory = input_directory

        output_def = simulation_def.get('output')
        if output_def is not None:
            output_directory = output_dir if output_dir is not None else output_def.get('path', '')
        else:
            output_directory = ''
        if not os.path.isabs(output_directory):
            output_directory = os.path.join(simulation_dir, output_directory)
        if not os.path.exists(output_directory):
            print("creating directory: '%s'" % output_directory)
            os.makedirs(output_directory)
        config.output_directory = output_directory

        if output_file is None:
            output_file = output_def.get('file')
            assert output_file is not None

        output_path = os.path.join(output_directory, output_file)

        entities = {}
        for k, v in content['entities'].iteritems():
            entities[k] = Entity.from_yaml(k, v)

        for entity in entities.itervalues():
            entity.attach_and_resolve_links(entities)

        global_context = {'__globals__': global_symbols(globals_def),
                          '__entities__': entities}
        parsing_context = global_context.copy()
        parsing_context.update((entity.name, entity.all_symbols(global_context))
                               for entity in entities.itervalues())
        for entity in entities.itervalues():
            parsing_context['__entity__'] = entity.name
            entity.parse_processes(parsing_context)
            entity.compute_lagged_fields()
            # entity.optimize_processes()

        if 'init' not in simulation_def and 'processes' not in simulation_def:
            raise SyntaxError("the 'simulation' section must have at least one "
                              "of 'processes' or 'init' subsection")
        # for entity in entities.itervalues():
        #     entity.resolve_method_calls()
        used_entities = set()
        init_def = [d.items()[0] for d in simulation_def.get('init', [])]
        init_processes = []
        for ent_name, proc_names in init_def:
            if ent_name not in entities:
                raise Exception("Entity '%s' not found" % ent_name)

            entity = entities[ent_name]
            used_entities.add(ent_name)
            init_processes.extend([(entity.processes[proc_name], 1)
                                   for proc_name in proc_names])

        processes_def = [d.items()[0]
                         for d in simulation_def.get('processes', [])]
        processes = []
        for ent_name, proc_defs in processes_def:
            if ent_name != 'legislation':
                entity = entities[ent_name]
                used_entities.add(ent_name)
                for proc_def in proc_defs:
                    # proc_def is simply a process name
                    if isinstance(proc_def, basestring):
                        # use the default periodicity of 1
                        proc_name, periodicity = proc_def, 1
                    else:
                        proc_name, periodicity = proc_def
                    processes.append((entity.processes[proc_name], periodicity))

        entities_list = sorted(entities.values(), key=lambda e: e.name)
        declared_entities = set(e.name for e in entities_list)
        unused_entities = declared_entities - used_entities
        if unused_entities:
            suffix = 'y' if len(unused_entities) == 1 else 'ies'
            print("WARNING: entit%s without any executed process:" % suffix,
                  ','.join(sorted(unused_entities)))

        if input_def is not None:
            input_method = input_def.get('method', 'h5')
        else:
            input_method = 'h5'

        input_path = os.path.join(input_directory, input_file)

        default_entity = simulation_def.get('default_entity')

        if runs is None:
            runs = simulation_def.get('runs', 1)
        return TilSimulation(
            globals_def,
            periods,
            start_period,
            init_processes,
            processes,
            entities_list,
            input_method,
            input_path,
            output_path,
            default_entity,
            runs = runs,
            legislation = legislation,
            final_stat = final_stat,
            config_log = config_log,
            uniform_weight = uniform_weight)

    @classmethod
    def from_yaml(cls, fpath,
                  input_dir=None, input_file=None,
                  output_dir=None, output_file=None,
                  start_period=None, periods=None, seed=None,
                  skip_shows=None, skip_timings=None, log_level=None,
                  assertions=None, autodump=None, autodiff=None,
                  runs=None, uniform_weight=None):
        with open(fpath) as f:
            return cls.from_str(f, os.path.dirname(os.path.abspath(fpath)),
                                input_dir, input_file,
                                output_dir, output_file,
                                start_period, periods, seed,
                                skip_shows, skip_timings, log_level,
                                assertions, autodump, autodiff,
                                runs, uniform_weight)

    def load(self):
        return timed(self.data_source.load, self.globals_def, self.entities_map)

    @property
    def entities_map(self):
        return {entity.name: entity for entity in self.entities}

    def run_single(self, run_console=False, run_num=None):
        start_time = time.time()

        input_dataset = timed(self.data_source.load,
                              self.globals_def,
                              self.entities_map)

        globals_data = input_dataset.get('globals')
        timed(self.data_sink.prepare, self.globals_def, self.entities_map,
              input_dataset, self.start_period - 1)

        print(" * building arrays for first simulated period")
        for ent_name, entity in self.entities_map.iteritems():
            print("    -", ent_name, "...", end=' ')
            # TODO: this whole process of merging all periods is very
            # opinionated and does not allow individuals to die/disappear
            # before the simulation starts. We couldn't for example,
            # take the output of one of our simulation and
            # re-simulate only some years in the middle, because the dead
            # would be brought back to life. In conclusion, it should be
            # optional.
            timed(entity.build_period_array, self.start_period - 1)
        print("done.")

        if config.autodump or config.autodiff:
            if config.autodump:
                fname, _ = config.autodump
                mode = 'w'
            else:  # config.autodiff
                fname, _ = config.autodiff
                mode = 'r'
            fpath = os.path.join(config.output_directory, fname)
            h5_autodump = tables.open_file(fpath, mode=mode)
            config.autodump_file = h5_autodump
        else:
            h5_autodump = None

        # tell numpy we do not want warnings for x/0 and 0/0
        np.seterr(divide='ignore', invalid='ignore')

        process_time = defaultdict(float)
        period_objects = {}
        eval_ctx = EvaluationContext(self, self.entities_map, globals_data)

        def simulate_period(period_idx, period, processes, entities,
                            init=False):
            period_start_time = time.time()

            # set current period
            eval_ctx.period = period

            if config.log_level in ("functions", "processes"):
                print()
            print("period", period,
                  end=" " if config.log_level == "periods" else "\n")
            if init and config.log_level in ("functions", "processes"):
                for entity in entities:
                    print("  * %s: %d individuals" % (entity.name,
                                                      len(entity.array)))
            else:
                if config.log_level in ("functions", "processes"):
                    print("- loading input data")
                    for entity in entities:
                        print("  *", entity.name, "...", end=' ')
                        timed(entity.load_period_data, period)
                        print("    -> %d individuals" % len(entity.array))
                else:
                    for entity in entities:
                        entity.load_period_data(period)
            for entity in entities:
                entity.array_period = period
                entity.array['period'] = period

            # Longitudinal
#            person_name = 'individus'
#            person = [x for x in entities if x.name == person_name][0]
#            var_id = person.array.columns['id']
#            # Init
#            use_longitudinal_after_init = any(
#                varname in self.longitudinal for varname in ['salaire_imposable', 'workstate']
#                )
#            if init:
#                for varname in ['salaire_imposable', 'workstate']:
#                    self.longitudinal[varname] = None
#                    var = person.array.columns[varname]
#                    fpath = self.data_source.input_path
#                    input_file = HDFStore(fpath, mode="r")
#                    if 'longitudinal' in input_file.root:
#                        input_longitudinal = input_file.root.longitudinal
#                        if varname in input_longitudinal:
#                            self.longitudinal[varname] = input_file['/longitudinal/' + varname]
#                            if period not in self.longitudinal[varname].columns:
#                                table = DataFrame({'id': var_id, period: var})
#                                self.longitudinal[varname] = self.longitudinal[varname].merge(
#                                    table, on='id', how='outer')
#                    if self.longitudinal[varname] is None:
#                        self.longitudinal[varname] = DataFrame({'id': var_id, period: var})
#
#            # maybe we have a get_entity or anything nicer than that # TODO: check
#            elif use_longitudinal_after_init:
#                for varname in ['salaire_imposable', 'workstate']:
#                    var = person.array.columns[varname]
#                    table = DataFrame({'id': var_id, period: var})
#                    if period in self.longitudinal[varname]:
#                        import pdb
#                        pdb.set_trace()
#                    self.longitudinal[varname] = self.longitudinal[varname].merge(table, on='id', how='outer')
#
#            eval_ctx.longitudinal = self.longitudinal

            if processes:
                num_processes = len(processes)
                for p_num, process_def in enumerate(processes, start=1):
                    process, periodicity = process_def

                    # set current entity
                    eval_ctx.entity_name = process.entity.name

                    if config.log_level in ("functions", "processes"):
                        print("- %d/%d" % (p_num, num_processes), process.name,
                              end=' ')
                        print("...", end=' ')
                    print(period_idx)
                    print(periodicity)
                    if period_idx % periodicity == 0:
                        elapsed, _ = gettime(process.run_guarded, eval_ctx)
                    else:
                        elapsed = 0
                        if config.log_level in ("functions", "processes"):
                            print("skipped (periodicity)")

                    process_time[process.name] += elapsed
                    if config.log_level in ("functions", "processes"):
                        if config.show_timings:
                            print("done (%s elapsed)." % time2str(elapsed))
                        else:
                            print("done.")
                    self.start_console(eval_ctx)

            if config.log_level in ("functions", "processes"):
                print("- storing period data")
                for entity in entities:
                    print("  *", entity.name, "...", end=' ')
                    timed(entity.store_period_data, period)
                    print("    -> %d individuals" % len(entity.array))
            else:
                for entity in entities:
                    entity.store_period_data(period)
#            print " - compressing period data"
#            for entity in entities:
#                print "  *", entity.name, "...",
#                for level in range(1, 10, 2):
#                    print "   %d:" % level,
#                    timed(entity.compress_period_data, level)
            period_objects[period] = sum(len(entity.array)
                                         for entity in entities)
            period_elapsed_time = time.time() - period_start_time
            if config.log_level in ("functions", "processes"):
                print("period %d" % period, end=' ')
            print("done", end=' ')
            if config.show_timings:
                print("(%s elapsed)" % time2str(period_elapsed_time), end="")
                if init:
                    print(".")
                else:
                    main_elapsed_time = time.time() - main_start_time
                    periods_done = period_idx + 1
                    remaining_periods = self.periods - periods_done
                    avg_time = main_elapsed_time / periods_done
                    # future_time = period_elapsed_time * 0.4 + avg_time * 0.6
                    remaining_time = avg_time * remaining_periods
                    print(" - estimated remaining time: %s."
                          % time2str(remaining_time))
            else:
                print()

        print("""
=====================
 starting simulation
=====================""")
        try:
            simulate_period(0, self.start_period - 1, self.init_processes,
                            self.entities, init=True)
            main_start_time = time.time()
            periods = range(self.start_period,
                            self.start_period + self.periods)
            for period_idx, period in enumerate(periods):
                simulate_period(period_idx, period,
                                self.processes, self.entities)

            total_objects = sum(period_objects[period] for period in periods)
            avg_objects = str(total_objects // self.periods) \
                if self.periods else 'N/A'
            main_elapsed_time = time.time() - main_start_time
            ind_per_sec = str(int(total_objects / main_elapsed_time)) \
                if main_elapsed_time else 'inf'

            print("""
==========================================
 simulation done
==========================================
 * %s elapsed
 * %s individuals on average
 * %s individuals/s/period on average
==========================================
""" % (time2str(time.time() - start_time), avg_objects, ind_per_sec))

            show_top_processes(process_time, 10)
#            if config.debug:
#                show_top_expr()

            if run_console:
                ent_name = self.default_entity
                if ent_name is None and len(eval_ctx.entities) == 1:
                    ent_name = eval_ctx.entities.keys()[0]
                # FIXME: fresh_data prevents the old (cloned) EvaluationContext
                # to be referenced from each EntityContext, which lead to period
                # being fixed to the last period of the simulation. This should
                # be fixed in EvaluationContext.copy but the proper fix breaks
                # stuff (see the comments there)
                console_ctx = eval_ctx.clone(fresh_data=True,
                                             entity_name=ent_name)
                c = console.Console(console_ctx)
                c.run()

        finally:
            self.close()
            if h5_autodump is not None:
                h5_autodump.close()

    def run(self, run_console=False):
        for i in range(int(self.runs)):
            self.run_single(run_console, i)

    def start_console(self, context):
        if self.stepbystep:
            c = console.Console(context)
            res = c.run(debugger=True)
            self.stepbystep = res == "step"

    def close(self):
        self.data_source.close()
        self.data_sink.close()


    def save_log(self, config_log):
        assert config_log is not None
        output_dir = os.path.dirname(self.data_sink.output_path)
        file_path = os.path.join(output_dir, 'config_log.yml')
        with open(file_path, 'w') as outfile:
            outfile.write(yaml.dump(config_log))

    def save_git_hash(self):
        output_dir = os.path.dirname(self.data_sink.output_path)
        file_path = os.path.join(output_dir, 'git_hash.txt')
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        til_git_hash = get_git_head_revision('Til-Core')
        til_base_model_git_hash = get_git_head_revision('Til-France')
        with open(file_path, "w") as text_file:
            text_file.write("""Running at {}
til: {}
til_base_model: {}""".format(time_stamp, til_git_hash, til_base_model_git_hash))
