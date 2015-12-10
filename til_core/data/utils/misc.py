# -*- coding:utf-8 -*-
from __future__ import print_function

'''
Created on 2 août 2013
@author: a.eidelman
'''

import logging
import numpy as np
from pandas import Series, DataFrame
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import pdb


log = logging.getLogger(__name__)


def recode(var_in, list_el, method, dtype=None):
    '''
    code une variable à partir d'une autre
    attention à la liste et à son ordre pour des méthode avec comparaison d'ordre
    '''
    if dtype is None:
        dtype1 = var_in.dtype
        # dtype1 = var_in.max()
    output = Series(index=var_in.index, dtype=dtype)
    for el in list_el:
        val_in = el[0]
        val_out = el[1]
        if method is 'geq':
            output[var_in >= val_in] = val_out
        if method is 'eq':
            output[var_in == val_in] = val_out
        if method is 'leq':
            output[var_in <= val_in] = val_out
        if method is 'lth':
            output[var_in < val_in] = val_out
        if method is 'gth':
            output[var_in > val_in] = val_out
        if method is 'isin':
            output[var_in.isin(val_in)] = val_out
    return output


def index_repeated(nb_rep):
    '''
    Fonction qui permet de numeroter les réplications. Si [A,B,C] sont répliqués 3,4 et 2 fois alors la fonction retourne
    [0,1,2,0,1,2,3,0,1] qui permet ensuite d'avoir
    [[A,A,A,B,B,B,B,C,C],[0,1,2,0,1,2,3,0,1]] et d'identifier les replications
    '''
    id_rep = np.arange(nb_rep.max())
    id_rep = as_strided(id_rep, shape=nb_rep.shape + id_rep.shape, strides=(0,) + id_rep.strides)
    return id_rep[id_rep < nb_rep[:, None]]


def replicate(table):
    columns_ini = table.columns
    dtypes_ini = table.dtypes
    nb_rep_table = np.asarray(table['nb_rep'], dtype=np.int64)
    table_exp = np.asarray(table).repeat(nb_rep_table, axis=0)

    table_exp = DataFrame(table_exp, columns = columns_ini, dtype = float)

    # change pour avoir les dtype initiaux malgré le passage par numpy
    for dtype in [np.int64, np.int32, np.int16, np.int8, np.float32, np.float16, np.float64]:
        var_type = dtypes_ini == dtype
        modif_types = dtypes_ini[var_type].index.tolist()
        table_exp[modif_types] = table_exp[modif_types].astype(dtype)

    table_exp['id_rep'] = index_repeated(nb_rep_table)
    table_exp['id_ini'] = table_exp['id']
    table_exp['id'] = table_exp.index
    return table_exp


def _MinType_col_int_pos(col):
    '''
    retourne le type minimal d'une serie d'entier positif
    on notera le -2 car on a deux valeurs prises par 0 et -1
    cela dit, on retire une puissance de deux pour tenir compte des négatifs
    je ne sais pas si on peut préciser qu'on code que des positifs.
    '''
    if max(abs(col)) < 2 ** 7 - 2:
        return col.astype(np.int8)
    elif max(abs(col)) < 2 ** 15 - 2:
        return col.astype(np.int16)
    elif max(abs(col)) < 2 ** 31 - 2:
        return col.astype(np.int32)
    else:
        return col.astype(np.int64)


def minimal_dtype(table):
    '''
    Try to give columns the minimal type using -1 for NaN value
    Variables with only two non null value are put into boolean asserting NaN value as False
    Minimal type for float is not searched (only integer)
    When integer has positive and negative value, there is no obvious default value for NaN values so nothing is done.
    '''
    assert isinstance(table, pd.DataFrame)
    modif = {'probleme': [], 'boolean': [], 'int_one_sign': [], 'other_int': [], 'float': [], 'object': []}
    for colname in table.columns:
        col = table[colname]
        if len(col.value_counts()) <= 1:
            # TODO: pour l'instant bug si la valeur de départ était -1
            col = col.fillna(value=-1)
            modif['probleme'].append(colname)
        if col.dtype == 'O':
            # log.info(colname," is an object, with a good dictionnary, we could transform it into integer")
            modif['object'].append(colname)
        if col.dtype != 'O':
            if len(col.value_counts()) == 2:
                min = col.min()
                col = col.fillna(value=min)
                col = col - int(min)
                # modif['boolean'].append(colname)
                # table[colname] = col.astype(np.bool)
                table[colname] = col.astype(np.int8)
            else:
                try:
                    if (col[col.notnull()].astype(int) == col[col.notnull()]).all():
                        try:
                            col.loc[col.notnull()] = col[col.notnull()].astype(int).values
                        except:
                            # dans ce cas, col est déjà un int et même plus petit que int32
                            pass
                        if col.min() >= 0 or col.max() <= 0:  # un seul signe pour les valeurs
                            sign = 1 - 2 * (max(col) < 0)
                            col = col.fillna(value = -1 * sign)
                            modif['int_one_sign'].append(colname)
                            table[colname] = _MinType_col_int_pos(col)
                        else:
                            modif['other_int'].append(colname)
                    else:
                        if (col.isnull().any()):
                            modif['float'].append(colname)
                        else:
                            # TODO
                            modif['float'].append(colname)
                except:
                    pdb.set_trace()
    if modif['object']:
        log.info('Object type columns have not been modified : \n {}'.format(modif['object']))
    if modif['float']:
        log.info('Float type columns have not been modified : \n  {}'.format(modif['float']))
    if modif['other_int']:
        log.info('Integer type columns with positive AND negative values have not been modified : \n {}'.format(
            modif['other_int']))
    if modif['probleme']:
        log.info('There is no much distinct values for following variables : \n {}'.format(modif['probleme']))
    if modif['boolean']:
        log.info('Note that these columns are transformed into boolean : \n {}'.format(modif['boolean']))
        log.info('Note also that in these cases, missing value are set to False')
    if modif['int_one_sign']:
        log.info('Dtype have been also optimized for : \n {}'.format(modif['int_one_sign']))
    log.info('Missing values were set to -1 (or +1 when only negative values)')

    return table


def count_dup(data, var):
    counts = data.groupby(var).size()
    df2 = pd.DataFrame(counts, columns = ['size'])
    var_rep = df2[df2.size > 1]
    if len(var_rep) != 0:
        print ("Nombre de valeurs apparaissant plusieurs fois pour  : " + str(len(var_rep)))
    return len(var_rep)


def drop_consecutive_row(data, var_dup):
    '''
    Remove a row if it's the same than the previous one for all
    variables in var_dup
    '''
    to_drop = False
    for var in var_dup:
        to_drop = to_drop | (data[var].shift(1) != data[var])

    data['block'] = (to_drop).astype(int).cumsum()
    data = data.drop_duplicates('block')
    data = data.drop('block', axis = 1)
    return data
