from scipy.fftpack import ss_diff
from scipy.special import rel_entr
from zipfile import ZipFile

import os
import subprocess
import torch
import torch.nn.functional as F

class ConstraintProgramming():
    RHYTHM = 'rhythm'
    PITCH = 'pitch'
    BASIC_JAVA_CMD = f'java -cp ../../../target/minicpbp-1.0.jar minicpbp.examples.'
    FILENAME_TOKEN_RHYTHM = f'token_rhythm.dat'

    def __init__(self, config, frame_per_bar):
        self.config = config
        self.frame_per_bar = frame_per_bar

        self.config[self.RHYTHM]['weight_variation']['rate'] = (self.config[self.RHYTHM]['weight_variation']['weight_max'] - self.config[self.RHYTHM]['weight_variation']['weight_min']) / ((self.frame_per_bar * self.config[self.RHYTHM]['weight_variation']['nb_bars_group']) - 1)
        self.config[self.PITCH]['weight_variation']['rate'] = (self.config[self.PITCH]['weight_variation']['weight_max'] - self.config[self.PITCH]['weight_variation']['weight_min']) / ((self.frame_per_bar * self.config[self.PITCH]['weight_variation']['nb_bars_group']) - 1)

        # minicpbp useful paths
        self.minicpbp_path = config['minicpbp_path']
        self.minicpbp_music_path = os.path.join(self.minicpbp_path, 'src', 'main', 'java', 'minicpbp', 'examples', 'data', 'MusicCP')
        self.minicpbp_working_dir = os.path.join(self.minicpbp_path, 'src', 'main', 'java')
    
    def save_rhythm_token(self, rhythm_tokens):
        with open(os.path.join(self.minicpbp_music_path, self.FILENAME_TOKEN_RHYTHM), 'w') as f:
            num_sample = rhythm_tokens.shape[0]
            for j in range(num_sample):
                f.write(' '.join(map(str, rhythm_tokens[j].tolist())))
                f.write('\n')
    
    def get_cp_rhythm_idx(self, rhythm_tokens, rhythm_output, epoch, i, device):
        return self._cpbp_java(rhythm_tokens, rhythm_output, epoch, i, device, True)
    
    def get_cp_pitch_probs(self, pitch_tokens, pitch_output, epoch, i, device):
        return self._cpbp_java(pitch_tokens, pitch_output, epoch, i, device, False)
    
    def _cpbp_java(self, tokens, output, epoch, i, device, cp_on_rhythm):
        ml_weight = self._get_weight(i, cp_on_rhythm)

        key = self.RHYTHM if cp_on_rhythm else self.PITCH
        i_title = f'{key}______i:{i}_______w:{ml_weight}_______'
        print(i_title)

        num_sample = tokens.shape[0]
        probs = F.softmax(output, dim=-1)
        filename = f'cp_{key}_{epoch}_{i}.dat'

        # create .dat files for java CP model (ML.txt and Token.txt can be used to debug)
        with open(os.path.join(self.minicpbp_music_path, filename), 'w') as f, open('ML.txt', 'a') as f2, open('Token.txt', 'a') as f3:
            f2.write(i_title + '\n')
            f3.write(i_title + '\n')
            for j in range(num_sample):
                f.write(' '.join(map(str, tokens[j].tolist()[:i])))
                f.write(' ' + ' '.join(map(str, probs[j].tolist())))
                f.write('\n')

                f2.write(' '.join(map(str, probs[j].tolist())))
                f2.write('\n')

                f3.write(' '.join(map(str, tokens[j].tolist()[:i])))
                f3.write('\n')
        
        # get belief propagation probs from minicpbp
        current_dir_backup = os.getcwd()
        os.chdir(self.minicpbp_working_dir)
        cmd = self._get_java_command(key, filename, num_sample, i, ml_weight)
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        os.chdir(current_dir_backup)

        if process.returncode != 0:
            raise Exception(f'Java MiniCPBP failed: {process.stderr}')

        # replace probs with new ones from belief propagation (Oracle.txt can be used to debug)
        with open(os.path.join(self.minicpbp_music_path, filename[:-4] + '_results.dat'), 'r') as f, open('Oracle.txt', 'a') as f2:
            f2.write(i_title + '\n')
            for j in range(num_sample):
                line = f.readline()
                probs_cp = torch.as_tensor(list(map(float, (line.split())))).to(device)
                probs[j] = probs_cp
                f2.write(line)

        # if sampling rhythm token, direclty get the token value
        if cp_on_rhythm:
            idx = torch.multinomial(probs, 1).squeeze()

            return idx

        # else return the distribution and model.py will sample from the top 5 probs
        return probs

    def _get_weight(self, i, cp_on_rhythm=True):
        key = self.RHYTHM if cp_on_rhythm else self.PITCH
        tech = self.config[key]['weight_variation']['technique']
        if tech == 'constant':
            return self._weight_constant(key)
        elif tech == 'linear_up_reset':
            return self._weight_linear_up_reset(i, key)
        elif tech == 'linear_down_reset':
            return self._weight_linear_down_reset(i, key)
        elif tech == 'manual':
            return self._weight_manual(i, key)
        elif tech == 'bar_down_reset':
            return self._weight_bar_down_reset(i, key)
        elif tech == 'token_down_reset':
            return self._weight_token_down_reset(i, key)
        elif tech == 'token_down':
            return self._weight_token_down(i, key)
        elif tech == 'onset_down':
            # The weight is handled in the minicpbp model.
            # It could/should possibly be handled in this code instead.
            return -1
        else:
            raise Exception(f'Not a valid weight variation technique on {key}')
    
    # weight is constant throughout the generation
    def _weight_constant(self, key):
        return self.config[key]['weight_variation']['ml_weight']

    # linearly going up (reset after nb_bars_group bars)
    def _weight_linear_up_reset(self, i, key):
        num_token = i % (self.frame_per_bar * self.config[key]['weight_variation']['nb_bars_group'])
        ml_weight = self.config[key]['weight_variation']['weight_min'] + self.config[key]['weight_variation']['rate'] * num_token
        return ml_weight
    
    # linearly going down (reset after nb_bars_group bars)
    def _weight_linear_down_reset(self, i, key):
        num_token = i % (self.frame_per_bar * self.config[key]['weight_variation']['nb_bars_group'])
        ml_weight = self.config[key]['weight_variation']['weight_max'] - self.config[key]['weight_variation']['rate'] * num_token
        return ml_weight
    
    # weight of each bar is set manually in a list
    def _weight_manual(self, i, key):
        num_bar = (i // self.frame_per_bar) % self.config[key]['weight_variation']['nb_bars_group']
        ml_weight = self.config[key]['weight_variation']['weight_per_bar'][num_bar]
        return ml_weight
    
    # geometric decay after each bar (reset after nb_bars_group bars)
    def _weight_bar_down_reset(self, i, key):
        num_bar = (i // self.frame_per_bar) % self.config[key]['weight_variation']['nb_bars_group']
        ml_weight = self.config[key]['weight_variation']['weight_ratio'] ** num_bar
        return ml_weight
    
    # geometric decay after each token (reset after nb_bars_group bars)
    def _weight_token_down_reset(self, i, key):
        num_token = i % (self.frame_per_bar * self.config[key]['weight_variation']['nb_bars_group'])
        ml_weight = self.config[key]['weight_variation']['weight_ratio'] ** num_token
        return ml_weight
    
    # geometric decay after each token (after nb_bars_group bars, the weight is constant at 1.0)
    def _weight_token_down(self, i, key):
        out_of_spans = i >= (self.frame_per_bar * self.config[key]['weight_variation']['nb_bars_group'])
        ml_weight = 1.0 if out_of_spans else self.config[key]['weight_variation']['weight_ratio'] ** i
        return ml_weight
    
    def _get_java_command(self, key, filename, num_sample, i, ml_weight):
        return self._get_java_command_rhythm(filename, num_sample, i, ml_weight) if key is self.RHYTHM else self._get_java_command_pitch(filename, num_sample, i, ml_weight)
    
    def _get_java_command_rhythm(self, filename, num_sample, i, ml_weight):
        cp_model_name = self.config[self.RHYTHM]['model']['name']
        if cp_model_name == 'rhythmIncreasingReset':
            nb_bars_group = self.config[self.RHYTHM]['weight_variation']['nb_bars_group']
            return f'{self.BASIC_JAVA_CMD}{cp_model_name} {filename} {num_sample} {i} {ml_weight} {nb_bars_group}'.split()
        elif cp_model_name == 'rhythmAlldifferentReset':
            nb_bars_group = self.config[self.RHYTHM]['weight_variation']['nb_bars_group']
            return f'{self.BASIC_JAVA_CMD}{cp_model_name} {filename} {num_sample} {i} {ml_weight} {nb_bars_group}'.split()
        elif cp_model_name == 'rhythmAlldifferent':
            nb_bars_group = self.config[self.RHYTHM]['weight_variation']['nb_bars_group']
            return f'{self.BASIC_JAVA_CMD}{cp_model_name} {filename} {num_sample} {i} {ml_weight} {nb_bars_group}'.split()
        elif cp_model_name == 'rhythmAtleast':
            min_nb_notes = self.config[self.RHYTHM]['model']['min_nb_notes']
            nb_bars_group = self.config[self.RHYTHM]['weight_variation']['nb_bars_group']
            return f'{self.BASIC_JAVA_CMD}{cp_model_name} {filename} {num_sample} {i} {ml_weight} {min_nb_notes} {nb_bars_group}'.split()
        elif cp_model_name == 'rhythmAlldifferentLastbar':
            nb_bars_group = self.config[self.RHYTHM]['weight_variation']['nb_bars_group']
            return f'{self.BASIC_JAVA_CMD}{cp_model_name} {filename} {num_sample} {i} {ml_weight} {nb_bars_group}'.split()
        else:
            raise Exception(f'Not a valid cp model on {self.RHYTHM}')

    def _get_java_command_pitch(self, filename, num_sample, i, ml_weight):
        cp_model_name = self.config[self.PITCH]['model']['name']
        if cp_model_name == 'pitchKey':
            k = self.config[self.PITCH]['model']['k']
            nb_bars_group = self.config[self.PITCH]['weight_variation']['nb_bars_group']
            return f'{self.BASIC_JAVA_CMD}{cp_model_name} {filename} {num_sample} {i} {self.FILENAME_TOKEN_RHYTHM} {ml_weight} {k} {nb_bars_group}'.split()
        elif cp_model_name == 'pitchKeyOnset':
            k = self.config[self.PITCH]['model']['k']
            nb_bars_group = self.config[self.PITCH]['weight_variation']['nb_bars_group']
            min_weight = self.config[self.PITCH]['weight_variation']['weight_min']
            return f'{self.BASIC_JAVA_CMD}{cp_model_name} {filename} {num_sample} {i} {self.FILENAME_TOKEN_RHYTHM} {min_weight} {k} {nb_bars_group}'.split()
        else:
            raise Exception(f'Not a valid cp model on {self.PITCH}')
