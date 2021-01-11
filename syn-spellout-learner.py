# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import itertools, re
import itertools
import numpy as np
import pandas as pd
import random
import xarray as xr
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import sys

# =============================================================================
# # ===========================================================================
# # LEARNER SETTINGS
# # ===========================================================================
# =============================================================================

input_data = 'kida-kido-kidi'
multiplier = 30
learn_x_times = 1 #how many times to run the learner

verbose=True
debug=False
write=True
write_log=True

starting_weight = 10.

aggressive=True
aggressive_threshold=1.35

reward_generalizing=False
generalizing_reward=0.5
reward_rederived=False
rederived_reward=0.2
#n_positions = 5

input_data_dict = {
    'kida': '0A-kida.txt',
    'kida+dual': '0A\'-kida+dual.txt',
    'doertu': '0A\'-doertu.txt',
    'kida&girla': '0B-kida&girla.txt',
    'kida&tabla': '0B-kida&tabla.txt',
    'kida+anim': '0C-kida+anim.txt',
    'kida&girla&tabla': '0D-kida&girla&tabla.txt',
    'kida-kido-kidi': '0E-kida-kido-kidi.txt'#,
  }


def feature_vector_translator(featuresAsSet,masterFeatureList):
    translatedVector = []
    prevUnseenFeats = [observationFeat
                             for observationFeat in featuresAsSet
                             if observationFeat not in masterFeatureList]
    masterFeatureList = masterFeatureList + prevUnseenFeats
    for feature in masterFeatureList:
        translatedVector.append(int(feature in featuresAsSet))
    return translatedVector, masterFeatureList

def redirect_to_file(text):
    original = sys.stdout
    sys.stdout = open('/Users/naomilee/Dropbox/nl-code/syn-spellout-learner/Data/' 
                      + input_data + '-Results/' + 'Run-' + str(i) + '/LearningLog.txt', 'a')
    print(text)
    sys.stdout = original

# =============================================================================
# # ===========================================================================
# # LEARNING-RELATED FUNCTIONS
# # ===========================================================================
# =============================================================================


# =============================================================================
# Extends root_da with a new Root
# =============================================================================
def add_to_Encyclopedia(root_da, observation, masterFeatureList):
    newRoot = xr.DataArray(
        data=np.zeros((1,len(masterFeatureList)),np.int8),
        coords={
            'root': [observation[1]],
            'feature': masterFeatureList,
            'spellout': xr.DataArray(data=[observation[0]], dims=['root'])
        },
        dims=('root', 'feature')
    )  
    
    if verbose:
        print('\n Just added a new Root to the Encyclopedia: ' + newRoot['root'].values.item() + '\n')
        if write_log:
            redirect_to_file('\n Just added a new Root to the Encyclopedia: ' + newRoot['root'].values.item() + '\n')
    
    return (
        xr.concat([root_da, newRoot], dim='root')        
    )


# =============================================================================
# Stems Roots based on shared subtring across observations of each Root,
# storing updates to root_da
# =============================================================================
def root_update(observation, root_da):
    current_Root_spellout = root_da.sel(root=observation[1])['spellout'].values.item()

    #If the stored VI for the Root isn't a substring of the latest observation...    
    while current_Root_spellout not in observation[0]: 
        #Take off segments from right (only suffixing)
        current_Root_spellout = current_Root_spellout[:-1]
        
    #Update the root's spellout coordinate in root_da
    root_da['spellout'].loc[dict(root=observation[1])] = current_Root_spellout
    
    return root_da


# =============================================================================
# Finds existing VIs in VI_rule_da that expone a given feature vector
# also HELPER function to generalized_rule
# =============================================================================
def same_morpheme(feature_vector, VI_rule_da):    
    #Converts list of ints (feature_vector) into DataArray with a `feature' dimension, with the coordinates of rule_da
    feature_da = xr.DataArray(
        data=feature_vector, 
        dims=['feature'], 
        coords={'feature': VI_rule_da['feature']}
    )
    
    #Broadcasts that 1-row DataArray to the same size / coordinates as VI_rule_da
    broadcast_feature_da = (
        xr.broadcast(feature_da, VI_rule_da)
        [0]
        .transpose('spellout', 'feature')
    )
    
    #Returns subset of rule_da where the whole `feature' dimension in a given 'spellout' coord 
    #is the same as our test feature_vector
    return (
        VI_rule_da
        .where(
            (VI_rule_da == broadcast_feature_da)
            .all('feature'),
            drop=True
        )
    )


# =============================================================================
# Finds existing VIs in VI_rule_da that have a given spellout
# also HELPER function to new_VI_rule and generalized_rule
# =============================================================================
def same_pronunciation(string_to_test, VI_rule_da):    
    if string_to_test == '':
        same_spellOuts_list = [
            x
            for x 
            in VI_rule_da.coords['spellout'].values 
            if 'blank' == x.split('_')[0]
        ]
    
    else:
        same_spellOuts_list = [
            x
            for x
            in VI_rule_da.coords['spellout'].values 
            if string_to_test == x.split('_')[0]
        ]
    
    same_spellOuts_da = VI_rule_da.sel(spellout = same_spellOuts_list)
    
    return same_spellOuts_list, same_spellOuts_da


# =============================================================================
# Converts an affix string and vector of features, set at a baseline weight of 10.0, to a DataArray object
# Also a HELPER function to add_to_Vocabulary
# =============================================================================
def new_VI_rule(affixString, featureVector, VI_rule_da, masterFeatureList):
    if affixString == '':
        newSpellOutCoord = 'blank_' + str(len(same_pronunciation(affixString,VI_rule_da)[0])+1)

    else:
        newSpellOutCoord = affixString + '_' + str(len(same_pronunciation(affixString,VI_rule_da)[0])+1)

    newVI = xr.DataArray(
        data=[featureVector],
        coords={
            'spellout': [newSpellOutCoord],
            'feature': masterFeatureList,
            'weight': xr.DataArray(data=[starting_weight], dims=['spellout'])},
        dims = ('spellout', 'feature')
    )
    
    return newVI


# =============================================================================
# Extends VI_rule_da with a new VI
# =============================================================================
def add_to_Vocabulary(VI_rule_da, newVI, syntax_da):
    if np.array_str(newVI.isel(spellout=0).values) not in syntax_da['feature_vector']:        
        new_feature_vector_coord = xr.DataArray(
            data=[np.array_str(newVI.isel(spellout=0).values)],
            dims=['feature_vector']
        )
        
        new_feature_vector = xr.DataArray(
            data=np.array([0.0, 0.0, 0.0, 0.0, 0.0], ndmin=2),
            coords={
                'position': syntax_da['position'],
                'feature_vector': new_feature_vector_coord
            },
            dims=('feature_vector', 'position')
        )
        
        syntax_da = xr.concat([syntax_da, new_feature_vector], dim='feature_vector')
    
    if verbose:
        print('Just created a new VI pronounced: ' 
             + newVI['spellout'].values.item() + '\n' 
             + str(newVI.where(newVI == 1, drop=True)['feature'].values.tolist()) + '\n')
        if write_log:
            redirect_to_file('Just created a new VI pronounced: ' 
                             + newVI['spellout'].values.item() + '\n' 
                             + str(newVI.where(newVI == 1, drop=True)['feature'].values.tolist()) + '\n')
    
    return (
        xr.concat([VI_rule_da, newVI], dim='spellout'),
        syntax_da        
    )


# =============================================================================
# HELPER function to generalized_rule
# Checks if string1 and string2 share any letters not including _n
# =============================================================================
# TODO handle cases where match is in the middle of a string
def shared_substring(string1, string2):
    string1 = string1.split('_')[0]
    string2 = string2.split('_')[0]
    
    if string1 == 'blank':
        return ('blank', '', string2)
    
    if string2 == 'blank':
        return ('blank', string1, '')
    
    match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    
    substring = string1[match.a: match.a + match.size]
    
    return substring, string1.replace(substring, ''), string2.replace(substring, '')


# =============================================================================
# HELPER function to generalized_rule
# Finds VIs in VI_rule_da to compare with newVI, using following criteria:
# must intersect on >=1 feature, share some subtring (or one is null), is not itself
# =============================================================================
def compare_rules(newVI, VI_rule_da):
    #Finds entries in VI_rule_da that share at least one feature
    feature_match_subset = (
        xr.broadcast(newVI, VI_rule_da)
        [0]
        .ffill('spellout')
        .bfill('spellout')
        .pipe(lambda x: x * VI_rule_da)
        .sum('feature')
        .pipe(lambda x: x > 0)
    )
    
    #Of those that share at least one feature, 
    #selects spellouts of those that share a substring - but not itself
    spellout_match_list = [
        spellout
        for spellout
        in (
                feature_match_subset['spellout']
                .where(feature_match_subset, drop=True)
                .values
        )
        if (
                len(shared_substring(spellout, newVI['spellout'].values.item())[0]) > 0
                and 
                (spellout != newVI['spellout'].values.item())
           )
    ]
    
    #DataArray containing entries in VI_rule_da that share both >=1 feature and some phono content 
    spellout_match_subset = VI_rule_da.sel(spellout=spellout_match_list)
    
    if verbose:
        print('Now, existing VIs being compared for shared feature(s)/substring matches are: ' 
              + str([x for x in spellout_match_subset['spellout'].values]) + '\n')
        if write_log:
            redirect_to_file('Now, existing VIs being compared for shared feature(s)/substring matches are: ' 
                             + str([x for x in spellout_match_subset['spellout'].values]) + '\n')
    
    return spellout_match_subset



# =============================================================================
# CREATES MORE GENERAL VI RULES
# Finds more general VI rules, by searching for VIs with intersecting feature vectors 
# and whose pronunciations share some substring
# =============================================================================
def generalized_rule(newVI, spellout_match_subset, VI_rule_da, masterFeatureList, syntax_da):
    for spellout in spellout_match_subset['spellout']:
        if verbose:
            print('\n Comparing ' + newVI['spellout'].values.item() + ' ' + str(newVI.squeeze().values)
            + ' with ' + spellout['spellout'].values.item()+ ' '
            + str(VI_rule_da.sel(spellout=spellout).squeeze().values) + ":\n")
            if write_log:
                redirect_to_file('\n Comparing ' + newVI['spellout'].values.item() + ' ' + str(newVI.squeeze().values)
                                + ' with ' + spellout['spellout'].values.item()+ ' '
                                + str(VI_rule_da.sel(spellout=spellout).squeeze().values) + ":\n")

        # -------------------------------------------------------------------------
        # multiply newVI with other rule: overlapping feature & shared substring
        # -------------------------------------------------------------------------
        feature_vector = (
            xr.broadcast(newVI, spellout_match_subset.sel(spellout=spellout))
            [0]
            .ffill('spellout')
            .bfill('spellout')
            .pipe(lambda x: x * spellout_match_subset.sel(spellout=spellout))
            .squeeze()
            .values
        )
        
        if verbose:
            print("1.intersecting fv "+ str(feature_vector))
            if write_log:
                redirect_to_file("1.intersecting fv "+ str(feature_vector))
        
        affixString = shared_substring(
                string1=newVI['spellout'].values.item(),
                string2=spellout_match_subset['spellout'].sel(spellout=spellout).values.item()
        )[0]
        
        if verbose:
            print("1.shared substring: " + str(affixString))
            if write_log:
                redirect_to_file("1.shared substring: " + str(affixString))
        
        existing_VIs_for_featureVector = same_morpheme(feature_vector, VI_rule_da)
        if len(same_pronunciation(affixString, existing_VIs_for_featureVector)[0]) == 0:
            generalizedVI = new_VI_rule(affixString, feature_vector, VI_rule_da, masterFeatureList)
            if verbose:
                print("1.intersection is a new VI!")#+str(generalizedVI))
                if write_log:
                    redirect_to_file("1.intersection is a new VI!")
            VI_rule_da, syntax_da = add_to_Vocabulary(VI_rule_da, generalizedVI, syntax_da)
            
            if reward_generalizing:
                VI_rule_da['weight'] =  (
                    VI_rule_da['weight']
                    +
                    (
                        xr.ones_like(VI_rule_da['weight'])
                        .pipe(lambda x: x * generalizing_reward)
                        .isel(spellout=[-1])
                        .reindex_like(VI_rule_da['weight'])
                        .fillna(0)
                    )
                )
                if verbose:
                    print ('just added an extra ' + str(generalizing_reward) + ' in weight to ' 
                           + VI_rule_da.isel(spellout=[-1])['spellout'].values.item())
                    if write_log:
                        redirect_to_file('just added an extra ' + str(generalizing_reward) + ' in weight to ' 
                           + VI_rule_da.isel(spellout=[-1])['spellout'].values.item())
        else:
            if reward_rederived:
                VI_rule_da['weight'] =  (
                    VI_rule_da['weight']
                    +
                    (
                        xr.ones_like(VI_rule_da['weight'])
                        .pipe(lambda x: x * rederived_reward)
                        .where(VI_rule_da['spellout']==same_pronunciation(
                                affixString, 
                                existing_VIs_for_featureVector
                            )
                            [0][0],
                            other=0
                        )
                    )
                )


        # -------------------------------------------------------------------------
        # Subtract newVI from other rule 
        # -------------------------------------------------------------------------
        feature_vector = (
            xr.broadcast(newVI, spellout_match_subset.sel(spellout=spellout))
            [0]
            .ffill('spellout')
            .bfill('spellout')
            .pipe(lambda x: spellout_match_subset.sel(spellout=spellout) - x)
            .pipe(lambda x: x.where(x > -1, other=0))
            .squeeze()
            .values
        )
        
        if verbose:
            print("2.comparand - newVI fv "+ str(feature_vector))
            if write_log:
                redirect_to_file("2.comparand - newVI fv "+ str(feature_vector))
        
        affixString = shared_substring(
                string1=newVI['spellout'].values.item(),
                string2=spellout_match_subset['spellout'].sel(spellout=spellout).values.item()
        )[2]
        
        if verbose:
            print("2.comparand - newVI string: " + str(affixString))
            if write_log:
                redirect_to_file("2.comparand - newVI string: " + str(affixString))
        
        existing_VIs_for_featureVector = same_morpheme(feature_vector, VI_rule_da)
        if len(same_pronunciation(affixString, existing_VIs_for_featureVector)[0]) == 0:
            generalizedVI = new_VI_rule(affixString, feature_vector, VI_rule_da, masterFeatureList)
            if verbose:
                print("2.comparand - newVI is a new VI!")#+ str(generalizedVI))
                if write_log:
                    redirect_to_file("2.comparand - newVI is a new VI!")
            VI_rule_da, syntax_da = add_to_Vocabulary(VI_rule_da, generalizedVI, syntax_da)
            
            if reward_generalizing:
                VI_rule_da['weight'] =  (
                    VI_rule_da['weight']
                    +
                    (
                        xr.ones_like(VI_rule_da['weight'])
                        .pipe(lambda x: x * generalizing_reward)
                        .isel(spellout=[-1])
                        .reindex_like(VI_rule_da['weight'])
                        .fillna(0)
                    )
                )
                if verbose:
                    print ('just added an extra ' + str(generalizing_reward) + ' in weight to ' 
                           + VI_rule_da.isel(spellout=[-1])['spellout'].values.item())
                    if write_log:
                        redirect_to_file('just added an extra ' + str(generalizing_reward) + ' in weight to ' 
                           + VI_rule_da.isel(spellout=[-1])['spellout'].values.item())
        else:
            if reward_rederived:
                VI_rule_da['weight'] =  (
                    VI_rule_da['weight']
                    +
                    (
                        xr.ones_like(VI_rule_da['weight'])
                        .pipe(lambda x: x * rederived_reward)
                        .where(VI_rule_da['spellout']==same_pronunciation(
                                affixString, 
                                existing_VIs_for_featureVector
                            )
                            [0][0],
                            other=0
                        )
                    )
                )

        # -------------------------------------------------------------------------
        # Subtract other rule from newVI
        # -------------------------------------------------------------------------
        feature_vector = (
            xr.broadcast(newVI, spellout_match_subset.sel(spellout=spellout))
            [0]
            .ffill('spellout')
            .bfill('spellout')
            .pipe(lambda x: x - spellout_match_subset.sel(spellout=spellout))
            .pipe(lambda x: x.where(x > -1, other=0))
            .squeeze()
            .values
        )
        
        if verbose:
            print("3.newVI - comparand fv " + str(feature_vector))
            if write_log:
                redirect_to_file("3.newVI - comparand fv " + str(feature_vector))
        
        affixString = shared_substring(
                string1=newVI['spellout'].values.item(),
                string2=spellout_match_subset['spellout'].sel(spellout=spellout).values.item()
        )[1]
        
        if verbose:
            print("3.newVI - comparand string: " + str(affixString))
            if write_log:
                redirect_to_file("3.newVI - comparand string: " + str(affixString))
        
        existing_VIs_for_featureVector = same_morpheme(feature_vector, VI_rule_da)
        if len(same_pronunciation(affixString, existing_VIs_for_featureVector)[0]) == 0:
            generalizedVI = new_VI_rule(affixString, feature_vector, VI_rule_da, masterFeatureList)
            if verbose:
                print("3.newVI - comparand is a new VI!")#+ str(generalizedVI))
                if write_log:
                    redirect_to_file("3.newVI - comparand is a new VI!")
            VI_rule_da, syntax_da = add_to_Vocabulary(VI_rule_da, generalizedVI, syntax_da)
            
            if reward_generalizing:
                VI_rule_da['weight'] =  (
                    VI_rule_da['weight']
                    +
                    (
                        xr.ones_like(VI_rule_da['weight'])
                        .pipe(lambda x: x * generalizing_reward)
                        .isel(spellout=[-1])
                        .reindex_like(VI_rule_da['weight'])
                        .fillna(0)
                    )
                )
                if verbose:
                    print ('just added an extra ' + str(generalizing_reward) + ' in weight to ' 
                           + VI_rule_da.isel(spellout=[-1])['spellout'].values.item())
                    if write_log:
                        redirect_to_file('just added an extra ' + str(generalizing_reward) 
                            + ' in weight to ' 
                            + VI_rule_da.isel(spellout=[-1])['spellout'].values.item())
        else:
            if reward_rederived:
                VI_rule_da['weight'] =  (
                    VI_rule_da['weight']
                    +
                    (
                        xr.ones_like(VI_rule_da['weight'])
                        .pipe(lambda x: x * rederived_reward)
                        .where(VI_rule_da['spellout']==same_pronunciation(
                                affixString, 
                                existing_VIs_for_featureVector
                            )
                            [0][0],
                            other=0
                        )
                    )
                )

    return VI_rule_da, syntax_da



# =============================================================================
# SPELLOUT-DRIVEN AGGRESSIVE SEGMENTATION OF ROOTS
# Learns how to stem Roots that always appear with one Root-adjacent morpheme
# Checks all Roots for right-edge overlaps with VI spellouts
# Restems Root pronunciation if, among VIs bearing that spellout, there is either:
# - a VI that expones a feature(s) that do not appear in the context of that observation at all, OR 
# - a VI that expones precisely a feature(s) that has previously seemed to be extraneous to the derivation:
#   (i.e. when a morpheme exponing it was added, caused a previously surface-true derivation to fail)
# That VI must be relatively high-weighted: at least 125% the weight of the lowest weighted known VI
# =============================================================================
def aggressive_decompose(observation, root_da, VI_rule_da, remnant_features, derivations, masterFeatureList):
    spellout_match_tuples = [
        (spellout.split("_")[0], spellout)
        for spellout 
        in VI_rule_da['spellout'].values
        if spellout.split("_")[0] != 'blank' and spellout.split("_")[0] == (
            root_da['spellout']
            .sel(root=observation[1])
            .values
            .item()
            [-len(spellout.split("_")[0]):]
        )
    ]
    
    spellout_remnant_overlap_match = (
        VI_rule_da
        .where(
            (
                VI_rule_da
                .sel(spellout=[spellout for (_, spellout) in spellout_match_tuples])
                .sel(feature=remnant_features)
                .sum('feature')
                .pipe(lambda x: x >= 1)
            ),
            drop=True
        )
        .where(
            (
                VI_rule_da
                .sel(spellout=[spellout for (_, spellout) in spellout_match_tuples])
                .sel(feature=[
                        feature 
                        for feature 
                        in VI_rule_da['feature'].values
                        if feature not in remnant_features
                    ]
                )
                .sum('feature')
                .pipe(lambda x: x == 0)
            ),
            drop=True
        )
    )
                
    if len(spellout_remnant_overlap_match['spellout']) > 0:
        spellout_remnant_overlap_match_max = (
            spellout_remnant_overlap_match
            .pipe(lambda x: x.where(x['weight'] == x['weight'].max('spellout'), drop=True))
#            .pipe(lambda x: x.where(x['weight'] > (
#                        VI_rule_da['weight']
#                        .sel(spellout=[spellout for (_, spellout) in spellout_match_tuples])
#                        .max('spellout')
#                    ),
#                    drop=True
#                )
#            )
            .pipe(lambda x: x.where(x['weight'] > (
                        VI_rule_da['weight']
                        .min('spellout')
                        .pipe(lambda x: x * aggressive_threshold)
                    ),
                    drop=True
                )
            )
        )
    else:
        spellout_remnant_overlap_match_max = spellout_remnant_overlap_match
        
                
    if len(spellout_remnant_overlap_match_max['spellout']) > 0:
        if verbose:
            print('Restemmed the spellout for the Root ' + observation[1] + ' as ' + root_da['spellout']
            .loc[dict(root=observation[1])]
            .values
            .item()
            [:-len(spellout_remnant_overlap_match_max['spellout'].values.item().split("_")[0])])
            if write_log:
                redirect_to_file('Restemmed the spellout for the Root ' + observation[1] + ' as ' 
                                 + root_da['spellout']
                                 .loc[dict(root=observation[1])]
                                 .values
                                 .item()
                                 [:-len(spellout_remnant_overlap_match_max['spellout'].values.item().split("_")[0])])
            
        root_da['spellout'].loc[dict(root=observation[1])] = (
            root_da['spellout']
            .loc[dict(root=observation[1])]
            .values
            .item()
            [:-len(spellout_remnant_overlap_match_max['spellout'].values.item().split("_")[0])]
        )
    
        return root_da
    
    ###
    
    
    spellout_no_overlap_match = (
        VI_rule_da
        .where(VI_rule_da.sum('feature') != 0, drop=True)
        .where(
            (
                VI_rule_da
                .sel(spellout=[spellout for (_, spellout) in spellout_match_tuples])
                .pipe(lambda x: 
                    x 
                    * 
                    (
                        new_VI_rule('temp', observation[2], VI_rule_da, masterFeatureList)
                        .pipe(xr.broadcast, x)[0]
                        .ffill('spellout')
                        .bfill('spellout')
                    )
                )
                .sum('feature')
                .pipe(lambda x: x == 0)
            ),
            drop=True
        )
    )
                    
    if len(spellout_no_overlap_match['spellout']) > 0:
        spellout_no_overlap_match_max = (
            spellout_no_overlap_match
            .pipe(lambda x: x.where(x['weight'] == x['weight'].max('spellout'), drop=True))
#            .pipe(lambda x: x.where(x['weight'] > (
#                        VI_rule_da['weight']
#                        .sel(spellout=[spellout for (_, spellout) in spellout_match_tuples])
#                        .max('spellout')
#                    ),
#                    drop=True
#                )
#            )
            .pipe(lambda x: x.where(x['weight'] > (
                        VI_rule_da['weight']
                        .min('spellout')
                        .pipe(lambda x: x * aggressive_threshold)
                    ),
                    drop=True
                )
            )
        )
    else:
        spellout_no_overlap_match_max = spellout_no_overlap_match
                
    if len(spellout_no_overlap_match_max['spellout']) > 0:
        if verbose:
            print('Restemmed the spellout for the Root ' + observation[1] + ' as ' + root_da['spellout']
            .loc[dict(root=observation[1])]
            .values
            .item()
            [:-len(spellout_no_overlap_match_max['spellout'].values.item().split("_")[0])])
            if write_log:
                redirect_to_file('Restemmed the spellout for the Root ' + observation[1] + ' as ' 
                                 + root_da['spellout']
                                 .loc[dict(root=observation[1])]
                                 .values
                                 .item()
                                 [:-len(spellout_no_overlap_match_max['spellout'].values.item().split("_")[0])])
            
        root_da['spellout'].loc[dict(root=observation[1])] = (
            root_da['spellout']
            .loc[dict(root=observation[1])]
            .values
            .item()
            [:-len(spellout_no_overlap_match_max['spellout'].values.item().split("_")[0])]
        )
        
        if root_da.loc[dict(root=observation[1])].sum('feature') == 0:
            root_da.loc[dict(root=observation[1])] = (
                spellout_no_overlap_match_max
                .squeeze()
                .values
            )
        if verbose:
            print(root_da)
            if write_log:
                redirect_to_file(root_da)
    
        return root_da
    
    return root_da


        
# =============================================================================
# UPDATES VI RELIABILITY AND SYNTACTIC KNOWLEDGE USING TEST DERIVATIONS
# Runs test derivations using context feature_vector to try and derive observed pronunciation
# Rewards successful VIs and the ordering of those VIs that was successful
# Punishes VIs that were unsuccessful in any order
# =============================================================================
def test_phase(observation, root_da, VI_rule_da, syntax_da, masterFeatureList):
    #Context feature vector
    temp_vi = new_VI_rule('', observation[2], VI_rule_da, masterFeatureList)
    
    temp_vi = (
        temp_vi
        .pipe(lambda x: x + (
                root_da
                .sel(root=observation[1])
                .values
            )
        )
        .pipe(lambda x: x.where(x <= 1, other=1))
    )
    
    temp_array = (
        temp_vi
        .pipe(lambda x: x.where(x == 1, drop=True))
        ['feature']
        .values
        .tolist()
    )
    
    #Randomizes which feature in the context feature vector is looked for within VI_rule_da
    random.shuffle(temp_array)
    
    #VIs chosen for use in this test derivation
    spellouts = []
    #Features in the context feature vector that have been discharged by chosen VIs
    features_to_skip = []
    
    
    matches_input = False
    remnant_features = []
    
    for feature in temp_array:
        if feature not in features_to_skip:
            if debug:
                print (feature)
            
            #Slices down VI_rule_da to only those VIs containing the feature we're trying to discharge
            sliced_rule_da = (
                VI_rule_da
                .where(VI_rule_da.sel(feature=feature) == 1, drop=True)
                .where(
                    (
                        xr.broadcast(temp_vi, VI_rule_da)[0]
                        .bfill('spellout')
                        .ffill('spellout')
                        .pipe(lambda x: VI_rule_da - x)
                        .max('feature')
                        .pipe(lambda x: x < 1)
                    ),
                    drop=True
                ) 
            )
             
            #Picks one of the available VIs based on their current weights 
            spellout = random.choices(
                population=sliced_rule_da['spellout'].values.tolist(),
                weights=sliced_rule_da['weight'].values.tolist()
            )
            
            if verbose:
                print('picked ' + str(spellout)
                    + ' to pronounce the feature ' + feature)
                if write_log:
                    redirect_to_file('picked ' + str(spellout)
                        + ' to pronounce the feature ' + feature)
            
            spellouts = spellouts + spellout
            
            #Creates tuple with the actual spellout string for each spellout index into VI_rule_da
            spellout_tuples_raw = [(spellout.split('_')[0], spellout) for spellout in spellouts]
            
            #Deals with changing 'blank' to ''
            spellout_tuples = [
                (spellout_string, spellout)
                if spellout_string != 'blank'
                else ('', spellout)
                for (spellout_string, spellout)
                in spellout_tuples_raw
            ]
            
            #Permutes all possible affix string orderings, retaining in a list the ordering of the spellout indexes
            permutations = [
                (
                    ''.join(
                        [
                            spellout_string
                            for (spellout_string, _)
                            in permuted_tuples
                        ]
                    ), 
                    [
                        spellout
                        for (_, spellout)
                        in permuted_tuples   
                    ]
                )
                for permuted_tuples
                in itertools.permutations(spellout_tuples)
            ]
            
            #In case there are multiple derivations that work
            #(i.e. our chosen spellouts includes a blank among their ranks),
            #randomizes which order of those spellouts' VIs gets rewarded
            random.shuffle(permutations)
            
            #Adds the Root pronunciations to the derived affix strings
            derivations = [
                (
                    (
                        root_da
                        .sel(root=observation[1])['spellout']
                        .values
                        .item() + spellout_string
                    ),
                    spellout_order
                )
                for (spellout_string, spellout_order)
                in permutations
            ]
                    
            if (observation[0] not in [spellout_string for spellout_string, _ in derivations] and matches_input):
                remnant_features.append(feature)
                if verbose:
                    print('Adding ' + str(feature) + ' was not helpful!')
                    if write_log:
                        redirect_to_file('Adding ' + str(feature) + ' was not helpful!')
    
            matches_input = (observation[0] in [spellout_string for spellout_string, _ in derivations])
            
            if debug:
                print('So far, it is ' + str(matches_input) + ' that we\'ve reproduced the input')
                
            #Identifies the features that that VI was associated with, to drop from the context feature vector
            features_to_drop = (
                VI_rule_da
                .sel(spellout=spellout)
                .pipe(lambda x: x.where(x == 1, drop=True))
                ['feature']
                .values
                .tolist()
            )
            
            #Discharges the features that that VI was associated with from the context feature vector
            temp_vi.loc[dict(feature=features_to_drop)] = 0
            
            for x in features_to_drop:
                if debug:
                    print ('skipped: ' + x)
                features_to_skip.append(x)
                
            if debug:
                print(features_to_skip)
    
    if verbose:
        print('remnant features: ' + str(remnant_features))
        if write_log:
            redirect_to_file('remnant features: ' + str(remnant_features))
            
    #Creates tuple with the actual spellout string for each spellout index into VI_rule_da
    spellout_tuples_raw = [(spellout.split('_')[0], spellout) for spellout in spellouts]
    
    #Deals with changing 'blank' to ''
    spellout_tuples = [
        (spellout_string, spellout)
        if spellout_string != 'blank'
        else ('', spellout)
        for (spellout_string, spellout)
        in spellout_tuples_raw
    ]
    
    #Permutes all possible affix string orderings, retaining in a list the ordering of the spellout indexes
    permutations = [
        (
            ''.join(
                [
                    spellout_string
                    for (spellout_string, _)
                    in permuted_tuples
                ]
            ), 
            [
                spellout
                for (_, spellout)
                in permuted_tuples   
            ]
        )
        for permuted_tuples
        in itertools.permutations(spellout_tuples)
    ]
    
    #In case there are multiple derivations that work
    #(i.e. our chosen spellouts includes a blank among their ranks),
    #randomizes which order of those spellouts' VIs gets rewarded
    random.shuffle(permutations)
    
    #Adds the Root pronunciations to the derived affix strings
    derivations = [
        (
            (
                root_da
                .sel(root=observation[1])['spellout']
                .values
                .item() + spellout_string
            ),
            spellout_order
        )
        for (spellout_string, spellout_order)
        in permutations
    ]
    
    if verbose:
        print('Tried to pronounce ' + observation[0] 
            + ' as ' + str([derived_form for (derived_form, _) in derivations]))
        if write_log:
            redirect_to_file('Tried to pronounce ' + observation[0] 
                + ' as ' + str([derived_form for (derived_form, _) in derivations]))
    
    if aggressive:
        root_da = aggressive_decompose(observation, root_da, VI_rule_da, remnant_features, derivations, masterFeatureList)
    
    #Look through all derivations...
    for (spellout_string, spellout_order) in derivations: 
        #checking if there was some successful derivation amongst the possible orderings of our chosen VIs...
        if observation[0] == spellout_string:
            #If so, add to the VI weights of all VIs involved
            VI_rule_da['weight'] =  (
                VI_rule_da['weight']
                +
                (
                    xr.ones_like(VI_rule_da['weight'])
                    .pipe(lambda x: x * .2)
                    .loc[dict(spellout=spellouts)]
                    .reindex_like(VI_rule_da['weight'])
                    .fillna(0)
                )
            )
            
            #...and add to the morpheme weights in involved positions
            for position, spellout in enumerate(spellout_order, start=1):
                syntax_da = (
                    syntax_da
                    +
                    (
                        xr.ones_like(syntax_da)
                        .pipe(lambda x: x * .2)
                        .pipe(lambda x: x.where(x['position']==position, other=0))
                        .pipe(lambda x: x.where(x['feature_vector']==(
                                    np.array_str(
                                        VI_rule_da
                                        .sel(spellout=spellout)
                                        .values
                                    )
                                ),
                                other=0
                            )
                        )
                    )    
                )
            
            return VI_rule_da, syntax_da, root_da

            
    #If we never found a single correct derivations in derivations, punish involved VIs     
    VI_rule_da['weight'] =  (
        VI_rule_da['weight']
        +
        (
            xr.ones_like(VI_rule_da['weight'])
            .pipe(lambda x: x * -.2)
            .loc[dict(spellout=spellouts)]
            .reindex_like(VI_rule_da['weight'])
            .fillna(0)
        )
    )

    return VI_rule_da, syntax_da, root_da



# =============================================================================
# # ===========================================================================
# # RUNS DERIVATIONS USING CURRENT GRAMMAR
# # Tries to derive syntax and pronunciation given a Root plus contextual features
# # Records number of times each surface string is derived,
# # plus number of times specific orders of specific VIs are used
# # Returns a dictionary summarizing results,
# # writes to [target-pronunciation]-[number-of-runs]-derivations.csv
# # ===========================================================================
# =============================================================================
def use_grammar(observation, loops, VI_rule_da, syntax_da, root_da, masterFeatureList):
    
    #Summary containing counts of resulting pronunciations, with syntactic rep of VIs in linear order used
    derivations_summary = {}
    
    if verbose:
        if write_log:
            redirect_to_file('Running ' + str(loops) + ' test derivations of ' + observation[0])
    
    for _ in itertools.repeat(None, loops):
        
        #Context feature vector
        context_features = new_VI_rule('', observation[2], VI_rule_da, masterFeatureList)
    
        #Add whatever syntactic features lexically select this Root and aren't present in context 
        #(i.e. inanimates' gender features)
        context_features = (
            context_features
            .pipe(lambda x: x + (
                    root_da
                    .sel(root=observation[1])
                    .values
                )
            )
            .pipe(lambda x: x.where(x <= 1, other=1))
        )
            
        context_features_array = (
            context_features
            .pipe(lambda x: x.where(x == 1, drop=True))
            ['feature']
            .values
            .tolist()
        )
        
        #Randomizes which feature in the context feature vector is looked for within VI_rule_da
        random.shuffle(context_features_array)
        
        #VIs chosen for use in this derivation
        spellouts = []
        #Features in the context feature vector that have been discharged by chosen VIs
        features_to_skip = []
        
        for feature in context_features_array:
            if feature not in features_to_skip:
                if debug:
                    print (feature)
                
                #Slices down VI_rule_da to only those VIs containing the feature we're trying to discharge
                sliced_rule_da = (
                    VI_rule_da
                    .where(VI_rule_da.sel(feature=feature) == 1, drop=True)
                    .where(
                        (
                            xr.broadcast(context_features, VI_rule_da)[0]
                            .bfill('spellout')
                            .ffill('spellout')
                            .pipe(lambda x: VI_rule_da - x)
                            .max('feature')
                            .pipe(lambda x: x < 1)
                        ),
                        drop=True
                    ) 
                )
                 
                #Picks one of the available VIs based on their current weights 
                spellout = random.choices(
                    population=sliced_rule_da['spellout'].values.tolist(),
                    weights=sliced_rule_da['weight'].values.tolist()
                )
                
                if verbose:
                    print('picked ' + str(spellout)
                    + ' to pronounce the feature ' + feature)
                    if write_log:
                        redirect_to_file('picked ' + str(spellout)
                        + ' to pronounce the feature ' + feature)
                
                spellouts = spellouts + spellout
                
                #Creates tuple with the actual spellout string for each spellout index into VI_rule_da
                spellout_tuples_raw = [(spellout.split('_')[0], spellout) for spellout in spellouts]
                
                #Deals with changing 'blank' to ''
                spellout_tuples = [
                    (spellout_string, spellout)
                    if spellout_string != 'blank'
                    else ('', spellout)
                    for (spellout_string, spellout)
                    in spellout_tuples_raw
                ]
                
                #Identifies the features that that VI was associated with, to drop from the context feature vector
                features_to_drop = (
                    VI_rule_da
                    .sel(spellout=spellout)
                    .pipe(lambda x: x.where(x == 1, drop=True))
                    ['feature']
                    .values
                    .tolist()
                )
                
                #Discharges the features that that VI was associated with from the context feature vector
                context_features.loc[dict(feature=features_to_drop)] = 0
                
                for x in features_to_drop:
                    if debug:
                        print ('skipped: ' + x)
                    features_to_skip.append(x)
                    
                if debug:
                    print(features_to_skip)
                    
        #Creates tuple with the actual spellout string for each spellout index into VI_rule_da
        spellout_tuples_raw = [(spellout.split('_')[0], spellout) for spellout in spellouts]
        
        #Deals with changing 'blank' to ''
        spellout_tuples = [
            (spellout_string, spellout)
            if spellout_string != 'blank'
            else ('', spellout)
            for (spellout_string, spellout)
            in spellout_tuples_raw
        ]
        
        if debug:
            print(spellout_tuples)
        
        if len(spellout_tuples) == 1:
            derivation_n_as_list = [spellout_tuples[0][1]]
            derivation = (
                root_da
                .sel(root=observation[1])['spellout']
                .values
                .item()
            ) + spellout_tuples[0][0]
        
        else:
            num_VIs = len(spellout_tuples)
            derivation_as_list = [''] * num_VIs
            derivation_n_as_list = [''] * num_VIs
            for VI in spellout_tuples:
                if num_VIs == 1:
                    #while derivation_as_list.count('') != 1:
                    #   derivation_as_list.pop(len(derivation_as_list) - 1 - derivation_as_list[::-1].index(''))
                    derivation_as_list[derivation_as_list.index('')] = VI[0]
                    derivation_n_as_list[derivation_n_as_list.index('')] = VI[1]
                    
                    break
                    
                VI_pos = int(
                    syntax_da
                    .sel(feature_vector=np.array_str(VI_rule_da.sel(spellout=VI[1]).values))
                    .pipe(lambda x: x.where(x == x.max('position'), drop=True))
                    ['position']
                    .values
                )
                
                if VI_pos > len(derivation_as_list):
                    VI_pos = len(derivation_as_list)
                    
                derivation_as_list[VI_pos - 1] = VI[0]
                derivation_n_as_list[VI_pos - 1] = VI[1]
                num_VIs -= 1
                
            derivation = (
                root_da
                .sel(root=observation[1])['spellout']
                .values
                .item()
            ) + ''.join(derivation_as_list)
            
        VIs_used = str(derivation_n_as_list)
        
        if derivation in derivations_summary.keys():
            current_count, current_dict = derivations_summary[derivation]
            if VIs_used in current_dict.keys():
                current_dict[VIs_used] = current_dict[VIs_used] + 1
                derivations_summary[derivation] = (current_count + 1, current_dict)
            else:
                current_dict[VIs_used] = 1
                derivations_summary[derivation] = (current_count + 1, current_dict)
        else:
            derivations_summary[derivation] = (1, {VIs_used: 1})   
        
    if write:
        pd.DataFrame(derivations_summary).to_csv(str('/Users/naomilee/Dropbox/nl-code/syn-spellout-learner/Data/' 
                             + input_data + '-Results/Run-' + str(i) + '/' + observation[0] + '-' + str(loops) + '-' + 'derivations.csv'))
    
    return derivations_summary



# =============================================================================
# # ===========================================================================
# # MAIN LEARNING FUNCTION
# # Learns over [multiplier] repetitions of the [input_data] file
# # Returns learned Roots, VIs, and syntax in:
# # root_da: Root meanings with pronunciations and features that must appear with that Root (e.g. uninterpretable gender)
# # VI_rule_da: spellouts with weights and the features they expone (in a feature vector)
# # syntax_da: unique feature vectors with weights in the positions they have successfully appeared
# # ===========================================================================
# =============================================================================
def run(input_data, multiplier):
    
    # -------------------------------------------------------------------------
    # Read in and store learning data
    # -------------------------------------------------------------------------
    
    #Open and read in learning data text file
    input_data_string = str('/Users/naomilee/Dropbox/nl-code/syn-spellout-learner/Data/' 
        + input_data_dict[input_data] )
    with open(input_data_string,'r') as learningDataFile:
        learningDataString = learningDataFile.read().splitlines()
    
    #Initialize list of all feature values encountered in the input
    masterFeatureList = []
    
    #Save learning data as list of lists, with pronunciation string, ROOT-MEANING, 
    #and vector of feature values
    learningData = []
    for line in learningDataString:
        tokens = line.split('\t')
        featuresAsSet = set(token for token in tokens[2:])
        featureVector, masterFeatureList = feature_vector_translator(featuresAsSet, masterFeatureList)
        latestInput = [tokens[0], tokens[1], featureVector]
        learningData.append(latestInput)
    #Make sure every observations's feature value vector is of the same length
    for observation in learningData:
        while len(observation[2]) < len(masterFeatureList):
            observation[2].append(0)
    if verbose:
        print('\n Learning data observations have been stored')
        if write_log:
            redirect_to_file('\n Learning data observations have been stored')

        
    # -------------------------------------------------------------------------
    # Time to learn!
    # -------------------------------------------------------------------------

    learningData = learningData * multiplier

    #random.shuffle(learningData)   
    
    #Initialize dictionary of roots with first observation's Root
    #Roots are stored by all-caps root meaning, not by numbered index
    root_da = xr.DataArray(
        data=np.zeros((1,len(masterFeatureList)), np.int8), 
        coords={
            'root': [learningData[0][1]],
            'feature': masterFeatureList,
            'spellout': xr.DataArray(data=[learningData[0][0]], dims=['root'])
        }, 
        dims=('root', 'feature')
    )
    
    if verbose:
        print(root_da)
        if write_log:
            redirect_to_file(root_da)
        
    #Initialize xarray DataArray of morphemes and VIs with first observation's feature values as null
    VI_rule_da = xr.DataArray(
        [learningData[0][2]], 
        coords={
            'spellout': ['blank_1'],
            'feature': masterFeatureList,
            'weight': xr.DataArray(data=[starting_weight], dims=['spellout'])
        }, 
        dims=('spellout','feature')
    )
    
    #Initialize xarray DataArray of positions and unique featureVectors that have appeared there
    positions_coord = xr.DataArray(data=[1, 2, 3, 4, 5],
                                   dims='position')
    
    feature_vector_coord = xr.DataArray(
        data=[np.array_str(VI_rule_da.sel(spellout='blank_1').values)],
        dims=['feature_vector']
    )
    
    syntax_da = xr.DataArray(
        data=np.array([0.0, 0.0, 0.0, 0.0, 0.0], ndmin=2),
        coords={
            'position': positions_coord,
            'feature_vector': feature_vector_coord
        },
        dims = ('feature_vector','position')
    )
    
    #Process 2nd through final observations in the learning data
    weight_ds = xr.Dataset()
    for index, observation in enumerate(learningData[1:]):
        if verbose:
            print('\n---------------------------\n' + 'Currently processing observation of: ' 
                  + observation[0] + '\n')
            if write_log:
                redirect_to_file('\n---------------------------\n' + 'Currently processing observation of: ' 
                  + observation[0] + '\n')
        #If we've seen this Root before...
        if observation[1] in root_da['root']:
            #Check that our existing spell-out for that Root makes sense (i.e. is a substring of this latest pronunciation), updating as needed
            root_da = root_update(observation, root_da)
            if debug:
                print(root_da)
            #Get the leftover affix string from the observation, given the (possibly updated) current Root spell-out
            affixString = observation[0].replace(
                root_da.sel(root=observation[1])['spellout'].values.item(),
                ''
            )
            #If there isn't any entry for the set of features in the observation...
            existing_VIs_for_featureVector = same_morpheme(observation[2],VI_rule_da)
            if not same_pronunciation(affixString, existing_VIs_for_featureVector)[0]:
                newVI = new_VI_rule(affixString, observation[2], VI_rule_da, masterFeatureList)
                VI_rule_da, syntax_da = add_to_Vocabulary(VI_rule_da, newVI, syntax_da)
                spellout_match_subset = compare_rules(newVI,VI_rule_da)
                VI_rule_da, syntax_da = generalized_rule(
                    newVI, spellout_match_subset, VI_rule_da, 
                    masterFeatureList, syntax_da
                )
                
        else:
            #Create an entry in root_da with the all-caps meaning as 'root', and the full string as 'spellout'
            root_da = add_to_Encyclopedia(root_da, observation, masterFeatureList)
            
            #Check how many blanks are stored in the VI_rule_dict, and create an entry with 'blank_[next_n]' as key, 
            #...and the feature set as value
            existing_VIs_for_featureVector = same_morpheme(observation[2], VI_rule_da)
            if not same_pronunciation('',existing_VIs_for_featureVector)[0]:
                newVI = new_VI_rule('', observation[2], VI_rule_da, masterFeatureList)
                VI_rule_da, syntax_da = add_to_Vocabulary(VI_rule_da,newVI,syntax_da)
                spellout_match_subset = compare_rules(newVI,VI_rule_da)
                VI_rule_da, syntax_da = generalized_rule(
                    newVI, spellout_match_subset, VI_rule_da, 
                    masterFeatureList, syntax_da
                )
        
        VI_rule_da, syntax_da, root_da = test_phase(observation, root_da, VI_rule_da, syntax_da, masterFeatureList)
            
        weight_ds = (
            xr.broadcast(weight_ds, VI_rule_da['weight'])
            [0]
            .assign(**{str(index):VI_rule_da['weight']})
        )

    return root_da, VI_rule_da, syntax_da, weight_ds




for i in range(learn_x_times):
    # =============================================================================
    # RUNS LEARNER
    # =============================================================================    
    root_da, VI_rule_da, syntax_da, weight_ds = run(input_data, multiplier)

    # =============================================================================
    # SAVES & WRITES RESULTS TO FILE
    # =============================================================================
    #Writes results to [input_data]-Results folder as Vocabulary.csv, Root.csv, and syntax.csv
    #Writes log of learning process to same folder
    if write:
        VI_results = VI_rule_da.to_pandas().join(VI_rule_da['weight'].to_pandas().rename('weight'))
        VI_results.to_csv(str('/Users/naomilee/Dropbox/nl-code/syn-spellout-learner/Data/' 
                                 + input_data + '-Results/' + 'Run-' + str(i) + '/Vocabulary.csv'))
        
        root_results = root_da.to_pandas().join(root_da['spellout'].to_pandas().rename('spellout'))
        root_results.to_csv(str('/Users/naomilee/Dropbox/nl-code/syn-spellout-learner/Data/' 
                                 + input_data + '-Results/' + 'Run-' + str(i) + '/Root.csv')) 
        
        syntax_da.to_pandas().to_csv(str('/Users/naomilee/Dropbox/nl-code/syn-spellout-learner/Data/' 
                                 + input_data + '-Results/' + 'Run-' + str(i) + '/syntax.csv')) 
        
        weight_dtf = weight_ds.to_array('index').to_pandas()
        
        weight_dtf.to_csv(str('/Users/naomilee/Dropbox/nl-code/syn-spellout-learner/Data/' 
                                 + input_data + '-Results/' + 'Run-' + str(i) + '/Vocabulary-weight-trace.csv'))
        
        plot = weight_dtf.plot()
        fig = plot.get_figure()
        fig.savefig(str('/Users/naomilee/Dropbox/nl-code/syn-spellout-learner/Data/' 
                                 + input_data + '-Results/' + 'Run-' + str(i) + '/Vocabulary-weight-trace.pdf'))
        