CMT\_CPBP
=========

A modified version of CMT (Chord Conditioned Melody Transformer, <https://ieeexplore.ieee.org/abstract/document/9376975>) where arbitrary constraints can be imposed on the generated melodies using constraint programming (CP).

Requirements
------------

-   librosa\>=0.9.1
-   matplotlib\>=3.5.2
-   midi\>=0.2.3
-   music21\>=7.3.3
-   numpy\>=1.22.4
-   pretty\_midi\>=0.2.9
-   PyYAML\>=6.0
-   scikit\_learn\>=1.1.1
-   scipy\>=1.8.1
-   tensorboardX\>=2.5.1
-   tensorflow\>=2.9.1
-   torch\>=1.11.0
-   tqdm\>=4.64.0

File descriptions
-----------------

-   `mxl2midi.py` : converts .mxl dataset to two track .mid dataset
-   `preprocess.py` : makes instance pkl files from two track midi files
-   `hparams.yaml` : specifies hyperparameters and paths to load data or save results
-   `dataset.py` : loads preprocessed pkl data
-   `layers.py` : self attention block and relative multi-head attention layers
-   `model.py` : implementation of CMT
-   `cp.py` : implementation to call MiniCPBP from CMT
-   `loss.py` : defines loss functions
-   `trainer.py` : utilities for loading, training, sampling and saving models
-   `run.py` : main code to train CMT or to directly generate the melodies from a CMT checkpoint model
-   `generation_metrics.py` : generates the metrics calculated on the melodies
-   `constraints_checker.py` : checks that the melodies generated satisfy the constraints imposed

### CMT

The files modified from the original CMT code are: `preprocess.py`, `hparams.yaml`, `model.py`, `trainer.py` and `run.py`.
The new files from the original CMT code are: `mxl2midi.py`, `cp.py`, `generation_metrics.py` and `constraints_checker.py`.
CMT's original code: <https://github.com/ckycky3/CMT-pytorch>.

### MiniCPBP

The commit version of MiniCPBP is: `6bd0144997998f069c67b5f6feb61341fcba8b48`. However, it may be recommended to use the more recent one, refer to <https://github.com/PesantGilles/MiniCPBP>. The new files added in MiniCPBP are the new CP models in `minicpbp\src\main\java\minicpbp\examples\`: 
- `rhythmIncreasingReset.java`: Each bar has a strictly greater number of notes than the previous one. Constraint resets after x bars. 
- `rhythmAlldifferentReset.java`: Each bar has a different number of notes. Constraint resets after x bars.
- `rhythmAlldifferent.java`: Each bar has a different number of notes.  Constraint affects only the first span of x bars.
- `rhythmAtleast.java`: There are at least n notes in the first span of x bars.
- `rhythmAlldifferentLastbar.java`: Each bar has a different number of notes and the last bar of a span must have twice the number of notes than the second to last bar of the span.
- `pitchKey.java`: Each pitch of the C major (or A minor) scale must appear at least k times. The decay of the oracle constraint's weight is happening after each token.
- `pitchKeyOnset.java`: Each pitch of the C major (or A minor) scale must appear at least k times. The decay of the oracle constraint's weight is happening after each onset token.

The folder `minicpbp\src\main\java\minicpbp\examples\data\MusicCP\` is also used to store testing data for the CP models.

### MGEval framework

Some functions in `mgeval\core.py` from the MGEval framework has been modified so that the metrics were calculated on the melody track of the .mid file. MGEval's original code: <https://github.com/RichardYang40148/mgeval>.

### midi library for python 3

No changes have been made in the midi library for python 3 from the original code (folder `python3-midi-master\`). Github: <https://github.com/louisabraham/python3-midi>

Training CMT
------------

The checkpoint of CMT trained on the EWLD dataset and used in our experiments is given in `results\idx001\model\`. However, if you wish to train your own CMT model, check the original documentation of CMT: <https://github.com/ckycky3/CMT-pytorch>.

MiniCPBP
--------

Check the README in the `minicpbp\`

Implementing a new constraint on the generated melodies
-------------------------------------------------------

1.  Create a new CP model in `minicpbp\src\main\java\minicpbp\examples\` that models the constraints to be imposed on the melodies.

2.  Execute `mvn package` command in the `minicpbp\` folder

3.  Add the new CP model's name in the if-else of method `_get_java_command_rhythm` or `_get_java_command_pitch` with its specific arguments.

4.  To add another way of varying the oracle constraint's weight, add the new method implementing the variation in the if-else of method `_get_weight`.

5.  Set the correct options in `hparams.yaml` under the model.cp options. There are options for the rhythm and the pitch CP model.

    -   activate: Set to True if you wish to add constraints to the rhythm (or pitch) of the melodies. Otherwise, the generation will only be based on CMT's probabilities.
    -   model&#46;name: The name of the CP model in minicpbp to use.
    -   model.min\_nb\_notes: The minimum number of notes to have in the first span of x bars. This option is only relevant when the `rhythmAtleast` CP model is used.
    -   weight\_variation.technique: The technique to use to vary the oracle constraint's weight during the generation of the melodies.
		- constant: The weight is fixed to a given value during the generation.
		- linear_up_reset: The weight is linearly increasing. Resets after x bars.
		- linear_down_reset: The weight is linearly decreasing. Resets after x bars.
		- manual: The weight is manually specified for each constrained bar in a list.
		- bar_down_reset: The weight is geometrically decaying after each bar. Resets after x bars.
		- token_down_reset: The weight is geometrically decaying after each token. Resets after x bars.
		- token_down: The weight is geometrically decaying after each token. After the first span of x bars, the weight is set to 1.0.
		- onset_down: The weight is geometrically decaying after each onset token. The weight is handled during the CP model.
    -   weight\_variation.nb\_bars\_group: The number of bars that are constrained.
    -   weight\_variation.ml\_weight: The fixed weight to use when weight\_variation.technique is set to 'constant'.
    -   weight\_variation.weight\_ratio: The common ratio r to use when weight\_variation.technique is set to 'bar\_down\_reset', 'token\_down\_reset' or 'token\_down'.
    -   weight\_variation.weight\_max: The maximum value when the weight when weight\_variation.technique is set to 'linear\_up\_reset' or 'linear\_down\_reset'. The rate is calculated depending of the max and the min value given.
    -   weight\_variation.weight\_min: The minimum value when the weight when weight\_variation.technique is set to 'linear\_up\_reset' or 'linear\_down\_reset'. The rate is calculated depending of the max and the min value given. This parameter is also used to set the minimum weight to reach when weight\_variation.technique is set to 'onset_down'.
    -   weight\_variation.weight\_per\_bar: The list of weights manually specified to use at each bar when weight\_variation.technique is set to 'manual'.
    -   model.k: The number of times each pitch of the scale must occur. This option is only relevant when the `pitchKey` or the `pitchKeyOnset` CP model is used.

6.  Generate the melodies by executing `run.py`. Make sure to include the `--load_rhythm` and `--sample` options.
