#!/usr/bin/env python

import os, sys

from pyrosetta import *
from pyrosetta.rosetta import *

import numpy as np
from collections import OrderedDict
import time
import argparse
import subprocess
import time
import glob

import torch
import json

import util_protein_mpnn as mpnn_util

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent, 'include'))
from silent_tools import silent_tools

init( "-beta_nov16 -in:file:silent_struct_type binary -mute all" +
    " -use_terminal_residues true -mute basic.io.database core.scoring" )

def cmd(command, wait=True):
    the_command = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if (not wait):
        return
    the_stuff = the_command.communicate()
    return str( the_stuff[0]) + str(the_stuff[1] )

def range1( iterable ): return range( 1, iterable + 1 )

#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser()
script_dir = os.path.dirname(os.path.realpath(__file__))

# I/O Arguments
parser.add_argument( "-pdbdir", type=str, default="", help='The name of a directory of pdbs to run through the model' )
parser.add_argument( "-silent", type=str, default="", help='The name of a silent file to run through the model' )
parser.add_argument( "-outpdbdir", type=str, default="outputs", help='The directory to which the output PDB files will be written, used if the -pdbdir arg is active' )
parser.add_argument( "-outsilent", type=str, default="out.silent", help='The name of the silent file to which output structs will be written, used if the -silent arg is active' )
parser.add_argument( "-runlist", type=str, default='', help="The path of a list of pdb tags to run, only active when the -pdbdir arg is active (default: ''; Run all PDBs)" )
parser.add_argument( "-checkpoint_name", type=str, default='check.point', help="The name of a file where tags which have finished will be written (default: check.point)" )

parser.add_argument( "-debug", action="store_true", default=False, help='When active, errors will cause the script to crash and the error message to be printed out (default: False)')

# Design Arguments
parser.add_argument( "-relax_cycles", type=int, default=1, help="The number of relax cycles to perform on each structure (default: 1)" )
parser.add_argument( "-output_intermediates", action="store_true", help='Whether to write all intermediate sequences from the relax cycles to disk (default: False)' )
parser.add_argument( "-seqs_per_struct", type=int, default="1", help="The number of sequences to generate for each structure (default: 1)" )

# ProteinMPNN-Specific Arguments
parser.add_argument( "-checkpoint_path", type=str, default=os.path.join(script_dir, 'ProteinMPNN/vanilla_model_weights/v_48_020.pt'), help=f"The path to the ProteinMPNN weights you wish to use, default {os.path.join(script_dir, 'ProteinMPNN/vanilla_model_weights/v_48_020.pt')}")
parser.add_argument( "-temperature", type=float, default=0.000001, help='The sampling temperature to use when running ProteinMPNN (default: 0.000001)' )
parser.add_argument( "-augment_eps", type=float, default=0, help='The variance of random noise to add to the atomic coordinates (default 0)' )
parser.add_argument( "-protein_features", type=str, default='full', help='What type of protein features to input to ProteinMPNN (default: full)' )
parser.add_argument( "-omit_AAs", type=str, default='CX', help='A string of all residue types (one letter case-insensitive) that you would not like to use for design. Letters not corresponding to residue types will be ignored (default: CX)' )
parser.add_argument( "-bias_AA_jsonl", type=str, default='', help='The path to a JSON file containing a dictionary mapping residue one-letter names to the bias for that residue eg. {A: -1.1, F: 0.7} (default: ''; no bias)' )
parser.add_argument( "-num_connections", type=int, default=48, help='Number of neighbors each residue is connected to. Do not mess around with this argument unless you have a specific set of ProteinMPNN weights which expects a different number of connections. (default: 48)' )

args = parser.parse_args( sys.argv[1:] )

class sample_features():
    '''
    This is a struct which keeps all the features related to a single sample together
    '''

    def __init__(self, pose, tag):
        self.pose = pose
        self.tag = os.path.basename(tag).split('.')[0]
    
    def parse_fixed_res(self):
        '''
            Parse which residues are fixed from the residue remarks
        '''

        # Iterate over the residue positions in this pose and check if they are fixed
        fixed_list = []
        #print("testxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx3")
        endA = self.pose.split_by_chain()[1].total_residue()
        for resi in range1( endA ):
            reslabels = self.pose.pdb_info().get_reslabels( resi )

            if len(reslabels) == 0: continue

            if str(self.pose.pdb_info().get_reslabels( resi )[1]).strip() == 'FIXED':
                fixed_list.append( resi )

        # Get ordered unique chainIDs for the pose
        # The last chain (B) is the target and will have a fixed sequence
        self.chains = list( OrderedDict.fromkeys( [ self.pose.pdb_info().chain(i) for i in range1( self.pose.total_residue() ) ] ) )

        # Create the fixed res dict, this will be input to ProteinMPNN
        self.fixed_res = {
            self.chains[0]: fixed_list#,
            #self.chains[0]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97],
            #self.chains[1]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97],
            #self.chains[2]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97],
            #self.chains[1]: []
        }
        #print("testxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx3")

        #self.tied_res = [{self.chains[0]: [1], self.chains[1]: [1], self.chains[2]: [1]}, {self.chains[0]: [2], self.chains[1]: [2], self.chains[2]: [2]}, {self.chains[0]: [3], self.chains[1]: [3], self.chains[2]: [3]}, {self.chains[0]: [4], self.chains[1]: [4], self.chains[2]: [4]}, {self.chains[0]: [5], self.chains[1]: [5], self.chains[2]: [5]}, {self.chains[0]: [6], self.chains[1]: [6], self.chains[2]: [6]}, {self.chains[0]: [7], self.chains[1]: [7], self.chains[2]: [7]}, {self.chains[0]: [8], self.chains[1]: [8], self.chains[2]: [8]}, {self.chains[0]: [9], self.chains[1]: [9], self.chains[2]: [9]}, {self.chains[0]: [10], self.chains[1]: [10], self.chains[2]: [10]}, {self.chains[0]: [11], self.chains[1]: [11], self.chains[2]: [11]}, {self.chains[0]: [12], self.chains[1]: [12], self.chains[2]: [12]}, {self.chains[0]: [13], self.chains[1]: [13], self.chains[2]: [13]}, {self.chains[0]: [14], self.chains[1]: [14], self.chains[2]: [14]}, {self.chains[0]: [15], self.chains[1]: [15], self.chains[2]: [15]}, {self.chains[0]: [16], self.chains[1]: [16], self.chains[2]: [16]}, {self.chains[0]: [17], self.chains[1]: [17], self.chains[2]: [17]}, {self.chains[0]: [18], self.chains[1]: [18], self.chains[2]: [18]}, {self.chains[0]: [19], self.chains[1]: [19], self.chains[2]: [19]}, {self.chains[0]: [20], self.chains[1]: [20], self.chains[2]: [20]}, {self.chains[0]: [21], self.chains[1]: [21], self.chains[2]: [21]}, {self.chains[0]: [22], self.chains[1]: [22], self.chains[2]: [22]}, {self.chains[0]: [23], self.chains[1]: [23], self.chains[2]: [23]}, {self.chains[0]: [24], self.chains[1]: [24], self.chains[2]: [24]}, {self.chains[0]: [25], self.chains[1]: [25], self.chains[2]: [25]}, {self.chains[0]: [26], self.chains[1]: [26], self.chains[2]: [26]}, {self.chains[0]: [27], self.chains[1]: [27], self.chains[2]: [27]}, {self.chains[0]: [28], self.chains[1]: [28], self.chains[2]: [28]}, {self.chains[0]: [29], self.chains[1]: [29], self.chains[2]: [29]}, {self.chains[0]: [30], self.chains[1]: [30], self.chains[2]: [30]}, {self.chains[0]: [31], self.chains[1]: [31], self.chains[2]: [31]}, {self.chains[0]: [32], self.chains[1]: [32], self.chains[2]: [32]}, {self.chains[0]: [33], self.chains[1]: [33], self.chains[2]: [33]}, {self.chains[0]: [34], self.chains[1]: [34], self.chains[2]: [34]}, {self.chains[0]: [35], self.chains[1]: [35], self.chains[2]: [35]}, {self.chains[0]: [36], self.chains[1]: [36], self.chains[2]: [36]}, {self.chains[0]: [37], self.chains[1]: [37], self.chains[2]: [37]}, {self.chains[0]: [38], self.chains[1]: [38], self.chains[2]: [38]}, {self.chains[0]: [39], self.chains[1]: [39], self.chains[2]: [39]}, {self.chains[0]: [40], self.chains[1]: [40], self.chains[2]: [40]}, {self.chains[0]: [41], self.chains[1]: [41], self.chains[2]: [41]}, {self.chains[0]: [42], self.chains[1]: [42], self.chains[2]: [42]}, {self.chains[0]: [43], self.chains[1]: [43], self.chains[2]: [43]}, {self.chains[0]: [44], self.chains[1]: [44], self.chains[2]: [44]}, {self.chains[0]: [45], self.chains[1]: [45], self.chains[2]: [45]}, {self.chains[0]: [46], self.chains[1]: [46], self.chains[2]: [46]}, {self.chains[0]: [47], self.chains[1]: [47], self.chains[2]: [47]}, {self.chains[0]: [48], self.chains[1]: [48], self.chains[2]: [48]}, {self.chains[0]: [49], self.chains[1]: [49], self.chains[2]: [49]}, {self.chains[0]: [50], self.chains[1]: [50], self.chains[2]: [50]}, {self.chains[0]: [51], self.chains[1]: [51], self.chains[2]: [51]}, {self.chains[0]: [52], self.chains[1]: [52], self.chains[2]: [52]}, {self.chains[0]: [53], self.chains[1]: [53], self.chains[2]: [53]}, {self.chains[0]: [54], self.chains[1]: [54], self.chains[2]: [54]}, {self.chains[0]: [55], self.chains[1]: [55], self.chains[2]: [55]}, {self.chains[0]: [56], self.chains[1]: [56], self.chains[2]: [56]}, {self.chains[0]: [57], self.chains[1]: [57], self.chains[2]: [57]}, {self.chains[0]: [58], self.chains[1]: [58], self.chains[2]: [58]}, {self.chains[0]: [59], self.chains[1]: [59], self.chains[2]: [59]}, {self.chains[0]: [60], self.chains[1]: [60], self.chains[2]: [60]}, {self.chains[0]: [61], self.chains[1]: [61], self.chains[2]: [61]}, {self.chains[0]: [62], self.chains[1]: [62], self.chains[2]: [62]}, {self.chains[0]: [63], self.chains[1]: [63], self.chains[2]: [63]}, {self.chains[0]: [64], self.chains[1]: [64], self.chains[2]: [64]}, {self.chains[0]: [65], self.chains[1]: [65], self.chains[2]: [65]}, {self.chains[0]: [66], self.chains[1]: [66], self.chains[2]: [66]}, {self.chains[0]: [67], self.chains[1]: [67], self.chains[2]: [67]}, {self.chains[0]: [68], self.chains[1]: [68], self.chains[2]: [68]}, {self.chains[0]: [69], self.chains[1]: [69], self.chains[2]: [69]}, {self.chains[0]: [70], self.chains[1]: [70], self.chains[2]: [70]}, {self.chains[0]: [71], self.chains[1]: [71], self.chains[2]: [71]}, {self.chains[0]: [72], self.chains[1]: [72], self.chains[2]: [72]}, {self.chains[0]: [73], self.chains[1]: [73], self.chains[2]: [73]}, {self.chains[0]: [74], self.chains[1]: [74], self.chains[2]: [74]}, {self.chains[0]: [75], self.chains[1]: [75], self.chains[2]: [75]}, {self.chains[0]: [76], self.chains[1]: [76], self.chains[2]: [76]}, {self.chains[0]: [77], self.chains[1]: [77], self.chains[2]: [77]}, {self.chains[0]: [78], self.chains[1]: [78], self.chains[2]: [78]}, {self.chains[0]: [79], self.chains[1]: [79], self.chains[2]: [79]}, {self.chains[0]: [80], self.chains[1]: [80], self.chains[2]: [80]}, {self.chains[0]: [81], self.chains[1]: [81], self.chains[2]: [81]}, {self.chains[0]: [82], self.chains[1]: [82], self.chains[2]: [82]}, {self.chains[0]: [83], self.chains[1]: [83], self.chains[2]: [83]}, {self.chains[0]: [84], self.chains[1]: [84], self.chains[2]: [84]}, {self.chains[0]: [85], self.chains[1]: [85], self.chains[2]: [85]}, {self.chains[0]: [86], self.chains[1]: [86], self.chains[2]: [86]}, {self.chains[0]: [87], self.chains[1]: [87], self.chains[2]: [87]}, {self.chains[0]: [88], self.chains[1]: [88], self.chains[2]: [88]}, {self.chains[0]: [89], self.chains[1]: [89], self.chains[2]: [89]}, {self.chains[0]: [90], self.chains[1]: [90], self.chains[2]: [90]}, {self.chains[0]: [91], self.chains[1]: [91], self.chains[2]: [91]}, {self.chains[0]: [92], self.chains[1]: [92], self.chains[2]: [92]}, {self.chains[0]: [93], self.chains[1]: [93], self.chains[2]: [93]}, {self.chains[0]: [94], self.chains[1]: [94], self.chains[2]: [94]}, {self.chains[0]: [95], self.chains[1]: [95], self.chains[2]: [95]}, {self.chains[0]: [96], self.chains[1]: [96], self.chains[2]: [96]}, {self.chains[0]: [97], self.chains[1]: [97], self.chains[2]: [97]}, {self.chains[0]: [98], self.chains[1]: [98], self.chains[2]: [98]}, {self.chains[0]: [99], self.chains[1]: [99], self.chains[2]: [99]}, {self.chains[0]: [100], self.chains[1]: [100], self.chains[2]: [100]}, {self.chains[0]: [101], self.chains[1]: [101], self.chains[2]: [101]}, {self.chains[0]: [102], self.chains[1]: [102], self.chains[2]: [102]}, {self.chains[0]: [103], self.chains[1]: [103], self.chains[2]: [103]}, {self.chains[0]: [104], self.chains[1]: [104], self.chains[2]: [104]}, {self.chains[0]: [105], self.chains[1]: [105], self.chains[2]: [105]}, {self.chains[0]: [106], self.chains[1]: [106], self.chains[2]: [106]}, {self.chains[0]: [107], self.chains[1]: [107], self.chains[2]: [107]}, {self.chains[0]: [108], self.chains[1]: [108], self.chains[2]: [108]}, {self.chains[0]: [109], self.chains[1]: [109], self.chains[2]: [109]}, {self.chains[0]: [110], self.chains[1]: [110], self.chains[2]: [110]}, {self.chains[0]: [111], self.chains[1]: [111], self.chains[2]: [111]}]
            #self.chains[0]: fixed_list,
            #self.chains[0]: [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]}#,
        #    self.chains[1]: [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        #    self.chains[2]: [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        #    self.chains[3]: []


        #}
    
    def thread_mpnn_seq(self, binder_seq):
        '''
        Thread the binder sequence onto the pose being designed
        '''
        #print("testxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx3")
        rsd_set = self.pose.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )

        for resi, mut_to in enumerate( binder_seq ):
            resi += 1 # 1 indexing
            name3 = mpnn_util.aa_1_3[ mut_to ]
            new_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( name3 ) )
            self.pose.replace_residue( resi, new_res, True )

        #print("testxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx3")
    

class ProteinMPNN_runner():
    '''
    This class is designed to run the ProteinMPNN model on a single input. This class handles
    the loading of the model, the loading of the input data, the FastRelax cycles, and the processing of the output
    '''

    def __init__(self, args, struct_manager):
        self.struct_manager = struct_manager
        #print("testxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx3")
        if torch.cuda.is_available():
            print('Found GPU will run ProteinMPNN on GPU')
            self.device = "cuda:0"
        else:
            print('No GPU found, running ProteinMPNN on CPU')
            self.device = "cpu"
        #print("testxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx3")
        self.mpnn_model = mpnn_util.init_seq_optimize_model(
            self.device,
            hidden_dim=128,
            num_layers = 3,
            backbone_noise = args.augment_eps,
            num_connections = args.num_connections,
            checkpoint_path = args.checkpoint_path
        )

        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

        self.debug = args.debug

        self.temperature = args.temperature
        self.seqs_per_struct = args.seqs_per_struct
        self.omit_AAs = [ letter for letter in args.omit_AAs.upper() if letter in list(alphabet) ]

        # Parse AA bias settings from json
        if os.path.isfile(args.bias_AA_jsonl):
            print(f'Found AA bias json file at {args.bias_AA_jsonl}')
            with open(args.bias_AA_jsonl, 'r') as json_file:
                json_list = list(json_file)
            for json_str in json_list:
                bias_AA_dict = json.loads(json_str)

            self.bias_AAs_np = np.zeros(len(alphabet))
            for n, AA in enumerate(alphabet):
                if AA in list(bias_AA_dict.keys()):
                    self.bias_AAs_np[n] = bias_AA_dict[AA]
        else:
            # Not using AA bias
            self.bias_AAs_np = np.zeros(len(alphabet))

        # Configs for the FastRelax cycles
        xml = os.path.join(script_dir, 'RosettaFastRelaxUtil-mono.xml')
        objs = protocols.rosetta_scripts.XmlObjects.create_from_file(xml)

        self.FastRelax = objs.get_mover('FastRelax')

        self.relax_cycles = args.relax_cycles
        #print("testxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    def relax_pose(self, sample_feats):
        '''
        Run FastRelax on the current pose
        '''

        relaxT0 = time.time()

        print('Running FastRelax')
        #print('Running ak;asdkadkadkl;asd;kasdkl;ad')
        #print(sample_feats)
        #print('Running ak;asdkadkadkl;asd;kasdkl;ad')
        #print(sample_feats.pose)
        #self.FastRelax.apply(sample_feats.pose)

        self.FastRelax.apply(sample_feats.pose)

        print('Running ak;asdkadkadkl;asd;kasdkl;ad')
        print(f"Completed one cycle of FastRelax in {int(time.time()) - relaxT0} seconds")

    def sequence_optimize(self, sample_feats):
        mpnn_t0 = time.time()

        # Once we have figured out pose I/O without Rosetta this will be easy to swap in
        pdbfile = 'temp.pdb'
        sample_feats.pose.dump_pdb(pdbfile)

        feature_dict = mpnn_util.generate_seqopt_features(pdbfile, sample_feats.chains)

        os.remove(pdbfile)

        arg_dict = mpnn_util.set_default_args( self.seqs_per_struct, omit_AAs=self.omit_AAs )
        arg_dict['temperature'] = self.temperature

        masked_chains  = sample_feats.chains[:-1]
        #print(masked_chains)
        #print("testsdaksdakldaskladskls")

        #masked_chains  = []
        visible_chains = [sample_feats.chains[-1]]
        print(visible_chains)

        fixed_positions_dict = {pdbfile[:-len('.pdb')]: sample_feats.fixed_res}
        #tied_positions_dict = {pdbfile[:-len('.pdb')]: sample_feats.tied_res}

        if self.debug:
            print(f'Fixed positions dict: {fixed_positions_dict}')

        print(fixed_positions_dict) 
        #print(tied_positions_dict) 


        sequences = mpnn_util.generate_sequences(
            self.mpnn_model,
            self.device,
            feature_dict,
            arg_dict,
            masked_chains,
            visible_chains,
            bias_AAs_np=self.bias_AAs_np,
            fixed_positions_dict=fixed_positions_dict#,
            #tied_positions_dict=tied_positions_dict
        )
        
        if self.debug:
            print(f'Generated sequence(s): {sequences}') 

        #print(fixed_positions_dict) 
        #print(tied_positions_dict)
        print(sequences)  
        print( f"ProteinMPNN generated {len(sequences)} sequences in {int( time.time() - mpnn_t0 )} seconds" ) 

        return sequences

    def proteinmpnn(self, sample_feats):
        '''
        Run ProteinMPNN sequence optimization on the pose, this does not use FastRelax
        '''
        seqs_scores = self.sequence_optimize(sample_feats)
        #print("testxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx3")
        # Iterate though each seq score pair and thread the sequence onto the pose
        # Then write each pose to a pdb file
        prefix = f"{sample_feats.tag}_dldesign"
        for idx, (seq, score) in enumerate(seqs_scores): 
            sample_feats.thread_mpnn_seq(seq)

            outtag = f"{prefix}_{idx}"

            self.struct_manager.dump_pose(sample_feats.pose, outtag)

    def proteinmpnn_fastrelax(self, sample_feats):
        '''
        Run ProteinMPNN plus FastRelax on the pose being designed
        '''
        tot_t0 = time.time()
        design_counter = 0

        prefix = f"{sample_feats.tag}_dldesign"
        #print("testxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx2")

        for cycle in range(args.relax_cycles):

            seqs_scores = self.sequence_optimize(sample_feats)

            seq, mpnn_score = seqs_scores[0] # We know there is only one entry
            sample_feats.thread_mpnn_seq(seq)

            self.relax_pose(sample_feats)

            if args.output_intermediates:
                tag = f"{prefix}_0_cycle{cycle}"
                self.struct_manager.dump_pose(sample_feats.pose, tag)

        seqs_scores = self.sequence_optimize(sample_feats)
        #print(seqs_scores)

        seq, mpnn_score = seqs_scores[0] # We know there is only one entry
        sample_feats.thread_mpnn_seq(seq)

        tag = f"{prefix}_0_cycle{args.relax_cycles}"
        self.struct_manager.dump_pose(sample_feats.pose, tag)

    def run_model(self, tag, args):
        t0 = time.time()

        print(f"Attempting pose: {tag}")
        
        # Load the pose 
        pose = self.struct_manager.load_pose(tag)

        # Initialize the features
        #print(pose)
        #print("dkdkdkasklasdkl;asdlk;asdlk;daslk;daskldaskladsk;lasd")
        sample_feats = sample_features(pose, tag)

        # Parse the fixed residues from the pose remarks
        sample_feats.parse_fixed_res()

        # Now determine which type of run we are doing and execute it
        if args.relax_cycles > 0:
            if args.seqs_per_struct > 1:
                raise Exception('Cannot use --seqs_per_struct > 1 with --relax_cycles > 0')

            self.proteinmpnn_fastrelax(sample_feats)

        else:
            self.proteinmpnn(sample_feats)

        seconds = int(time.time() - t0)

        print( f"Struct: {pdb} reported success in {seconds} seconds" )

class StructManager():
    '''
    This class handles all of the input and output for the ProteinMPNN model. It deals with silent files vs. pdbs,
    checkpointing, and writing of outputs

    Note: This class could be moved to a separate file
    '''

    def __init__(self, args):
        self.args = args

        self.silent = False
        if not args.silent == '':
            self.silent = True

            self.struct_iterator = silent_tools.get_silent_index(args.silent)['tags']

            self.sfd_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
            self.sfd_in.read_file(args.silent)

            self.sfd_out = core.io.silent.SilentFileData(args.outsilent, False, False, "binary", core.io.silent.SilentFileOptions())

            self.outsilent = args.outsilent

        self.pdb = False
        if not args.pdbdir == '':
            self.pdb = True

            self.pdbdir    = args.pdbdir
            self.outpdbdir = args.outpdbdir

            self.struct_iterator = glob.glob(os.path.join(args.pdbdir, '*.pdb'))

            # Parse the runlist and determine which structures to process
            if args.runlist != '':
                with open(args.runlist, 'r') as f:
                    self.runlist = set([line.strip() for line in f])

                    # Filter the struct iterator to only include those in the runlist
                    self.struct_iterator = [struct for struct in self.struct_iterator if os.path.basename(struct).split('.')[0] in self.runlist]

                    print(f'After filtering by runlist, {len(self.struct_iterator)} structures remain')

        # Assert that either silent or pdb is true, but not both
        assert(self.silent ^ self.pdb), f'Both silent and pdb are set to {args.silent} and {args.pdb} respectively. Only one of these may be active at a time'

        # Setup checkpointing
        self.chkfn = args.checkpoint_name
        self.finished_structs = set()

        if os.path.isfile(self.chkfn):
            with open(self.chkfn, 'r') as f:
                for line in f:
                    self.finished_structs.add(line.strip())

    def record_checkpoint(self, tag):
        '''
        Record the fact that this tag has been processed.
        Write this tag to the list of finished structs
        '''
        with open(self.chkfn, 'a') as f:
            f.write(f'{tag}\n')

    def iterate(self):
        '''
        Iterate over the silent file or pdb directory and run the model on each structure
        '''
        #print("wqejdakdaskdaskldsa")
        # Iterate over the structs and for each, check that the struct has not already been processed
        for struct in self.struct_iterator:
            tag = os.path.basename(struct).split('.')[0]
            if tag in self.finished_structs:
                print(f'{tag} has already been processed. Skipping')
                continue

            yield struct
        #print("asdasdsawqejdakdaskdaskldsa")

    def dump_pose(self, pose, tag):
        '''
        Dump this pose to either a silent file or a pdb file depending on the input arguments
        '''
        if self.pdb:
            # If the outpdbdir does not exist, create it
            # If there are parents in the path that do not exist, create them as well
            if not os.path.exists(self.outpdbdir):
                os.makedirs(self.outpdbdir)

            pdbfile = os.path.join(self.outpdbdir, tag + '.pdb')
            pose.dump_pdb(pdbfile)
        
        if self.silent:
            struct = self.sfd_out.create_SilentStructOP()
            struct.fill_struct(pose, tag)

            self.sfd_out.add_structure(struct)
            self.sfd_out.write_silent_struct(struct, self.outsilent)

    def load_pose(self, tag):
        '''
        Load a pose from either a silent file or a pdb file depending on the input arguments
        '''

        if not self.pdb and not self.silent:
            raise Exception('Neither pdb nor silent is set to True. Cannot load pose')

        if self.pdb:
            pose = pose_from_pdb(tag)
        
        if self.silent:
            pose = Pose()
            self.sfd_in.get_structure(tag).fill_pose(pose)
        
        return pose


####################
####### Main #######
####################
#print("testxxxxxxxxxxxxxxxxxxxxx123")

struct_manager     = StructManager(args)
#print("testxxxxxxxxxxxxxxxxxxxxx123")
proteinmpnn_runner = ProteinMPNN_runner(args, struct_manager)
#print("testxxxxxxxxxxxxxxxxxxxxx12345678")

for pdb in struct_manager.iterate():
    #print("asdlasdlasdldsaltestxxxxxxxxxxxxxxxxxxxxx123")

    if args.debug: proteinmpnn_runner.run_model(pdb, args)

    else: # When not in debug mode the script will continue to run even when some poses fail
        t0 = time.time()

        try: proteinmpnn_runner.run_model(pdb, args)

        except KeyboardInterrupt: sys.exit( "Script killed by Control+C, exiting" )

        except:
            seconds = int(time.time() - t0)
            print( "Struct with tag %s failed in %i seconds with error: %s"%( pdb, seconds, sys.exc_info()[0] ) )

    # We are done with one pdb, record that we finished
    struct_manager.record_checkpoint(pdb)
    



