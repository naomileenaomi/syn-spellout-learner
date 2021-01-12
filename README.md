# syn-spellout-learner
This model learns the spellout and syntax (templatic positions) of morphosyntactic features through cross-situational generalization. The learner is gradual and online.

It has no innate assumptions about the syntax of features. Though it is Separationist, it assumes that spellout always targets terminal nodes, so that learners can take surface distributional facts at face value, and so create feature bundles and place them in syntactic positions. Any bundle of features is a possible morpheme, and any set of morphemes can live in a given syntactic terminal slot. 

The model takes as input a series of nominal data: an ordered list of the phonological pronunciation of the noun, the Root concept meaning, and a list of contextually available semantic features. For example, a toy datum might be as follows: <sibas, SIBLING, woman, many>. The learner uses cross-situational learning to posit an inventory of feature bundles, and their matches to phonological realizations, then learns a template for their syntactic positioning. It generalizes its form-meaning hypotheses and assigns them weights over time by using them in test derivations of recently heard forms; this learning process results in a grammar that can produce pronunciations of Root-feature combinations probabilistically. 

The text files in the Data folder are toy datasets to demonstrate the learner's ability to learn from different patterns of gender and number morphosyntax on nouns. In order to try the learner yourself, make the following choices in syn-spellout-learner.py:
  - First, replace all instances of the directory path '/Users/naomilee/Dropbox/nl-code/syn-spellout-learner/' with the path to the folder where you have downloaded the .py file and the Data folder
  - Choose the toy dataset you are using, by referencing the input_data_dict on line 43 and changing the value of input_data on line 23 to the correct key.
  - Set the number of repetitions of the data in your chosen toy dataset you want to provide the learner, by setting the value of multiplier on line 24.
  - Set the number of times you want to see the learner do its thing, by setting the value of learn_x_times on line 25.
  - The verbose, write, and write_log flags on lines 27-30 determine the output that will be produced and/or saved in each run of the learner.

The learner will produce a directory ([dataset-name]-Results) containing a subdirectory for each run of the learner (Run-[n]) within the Data folder. Each Run folder will contain:
  - Root.csv: a Root inventory, which gives the list of Root concepts and their pronunciations. A matrix of Roots and features has a value of 1 whenever a given Root contributes a feature consistently (e.g. gender feature with an inanimate Root)
  - Vocabulary.csv: the functional Vocabulary, which gives the list of Vocabulary Items that the learner segmented and their corresponding abstract feature bundles that the learner considered. The feature bundle is given as a set of binary values corresponding to an ordered list of all features. Each Vocabulary Item has a weight: the higher the weight, the more reliable it was at generating correct pronunciations for a given bundle of features. Vocabulary-weight-trace.csv gives the history of the Vocabulary Items' reliability weights over the learner's experience. Vocabulary-weight-trace.pdf represents those traces graphically. 
  - syntax.csv: a template for the positioning of the abstract feature bundles. The bundles (represented as vectors of binary values corresponding to the ordered list of features) are given weights for each position slot: the higher the weight in a given position, the more often placing the bundle in that position generated a correct pronunciation for a given bundle of features. 

Please feel free to reach out with questions or feedback. This project is part of ongoing research.
