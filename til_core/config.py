# -*- coding: utf-8 -*-


import configparser
import os
from xdg import BaseDirectory


default_config_files_directory = BaseDirectory.save_config_path('til-core')


class Config(configparser.SafeConfigParser):
    config_local_ini = None
    config_ini = None

    def __init__(self, config_files_directory = default_config_files_directory):
        configparser.SafeConfigParser.__init__(self)
        assert config_files_directory is not None

        config_ini = os.path.join(config_files_directory, 'config.ini')
        if os.path.exists(config_ini):
            self.config_ini = config_ini
        self.read([config_ini])

    def save(self):
        assert self.config_local_ini or self.config_ini, "configuration file paths are not defined"
        config_file = open(self.config_ini, 'w')
        self.write(config_file)
        config_file.close()
