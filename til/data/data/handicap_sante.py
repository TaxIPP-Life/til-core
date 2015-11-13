# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:05:59 2015

@author: benjello
"""

import logging
import os
import pkg_resources


from openfisca_survey_manager.survey_collections import SurveyCollection


log = logging.getLogger(__name__)

config_files_directory = os.path.join(
    pkg_resources.get_distribution('openfisca-survey-manager').location)


# load data
survey_collection = SurveyCollection.load(
    collection = 'hsi', config_files_directory = config_files_directory)

survey = survey_collection.get_survey('hsi_hsi_2009')
# CDATCO
# Ancienneté de la vie en couple
# CDATDC
# Année de décès du conjoint
# CDATSE
# Année de la séparation effective
famille_variables = [ 'ident_ind', 'poids_hsi', 'sexe', 'etatmatri']
individus_variables = ['age', 'ident_ind', 'poids_hsi', 'sexe', 'etatmatri']
revenus_alloc_reconn_variables = ['ident_ind', 'rgir']
# AGFINETU
# Age de fin d'études initiales
# DIP14
# Diplôme le plus élevé (code regroupé)
scolarite_variables = ['agfinetu', 'dip14', 'ident_ind']

individus = survey.get_values(table = 'individus', variables = individus_variables)
revenus_alloc_reconn = survey.get_values(table = 'l_revenus_alloc_reconn', variables = revenus_alloc_reconn_variables)
scolarite = survey.get_values(table = 'j_scolarite', variables = scolarite_variables)

df = individus.merge(revenus_alloc_reconn)

z = df[df.age >= 60].pivot_table(values = 'poids_hsi', aggfunc = 'sum', index = 'age', columns = 'rgir')

age_en_mois
migrant
naiss
partner
tuteur
dur_in_couple
dur_out_couple
# etatmatri for civilstate  # MARRIED: 1, SINGLE: 2, DIVORCED: 3, WIDOW: 4, PACS: 5

# education
education_level
findet
