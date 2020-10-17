import pypdb
import random
import os
import sys
import subprocess
import re
import traceback
import shutil
import argparse

def create_file_set(p):
    
    print(' Pulling ' + p)

    # Get PDB File
    pdb_file = pypdb.get_pdb_file(p)

    pdb_file = modify_pdb(pdb_file)

    # Save PDB File
    f = open(target + '/' + p + '/DATA/' + p + '.pdb', 'w')
    f.write(pdb_file)
    print('\tWriting ' + p + ' PDB File')
    f.close()

    contents = subprocess.check_output('perl gen_sop_input.pl -i ' + target + '/' + p + '/DATA/' + p + '.pdb -b -a -v -x', shell=True, stderr=subprocess.STDOUT).decode('utf-8')
    match = re.search('(nbnd \d+\n(?:\d+ \d+ -?\d+\.\d+\n)+)(nangle \d+\n(?:\d+ \d+ \d+ -?\d+\.\d+\n)+)(natt \d+ nrep \d+ epsilon_h -?\d+\.\d+ epsilon_l -?\d+\.\d+ sigma -?\d+\.\d+ rcut_rna -?\d+\.\d+ rcut_prot -?\d+\.\d+\n(?:\d+ \d+ -?\d+\.\d+ \d+ \d+\n)+)(nbead \d+\n(?:\d+ -?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+ \d+\n)+)', contents)

    # Calculate and Save Bonds File
    f = open(target + '/' + p + '/DATA/' + p + '_bonds.dat', 'w+')
    f.write(match.group(1))
    print('\tWriting ' + p + ' Bonds File')
    f.close()

    # Calculate and Save Angles File
    f = open(target + '/' + p + '/DATA/' + p + '_angles.dat', 'w+')
    f.write(match.group(2))
    print('\tWriting ' + p + ' Angles File')
    f.close()

    # Calculate and Save VDW File
    f = open(target + '/' + p + '/DATA/' + p + '_vdw.dat', 'w+')
    f.write(match.group(3))
    print('\tWriting ' + p + ' VDW File')
    f.close()

    # Calculate and Save Init File
    f = open(target + '/' + p + '/DATA/' + p + '_init.xyz', 'w+')
    f.write(match.group(4))
    print('\tWriting ' + p + ' Init File')
    f.close()

def pull_and_generate(p, target):
    try:
        # Create Directories

        if not os.path.isdir(target + '/' + p):
            os.mkdir(target + '/' + p)

        if not os.path.isdir(target + '/' + p + '/DATA'):
            os.mkdir(target + '/' + p + '/DATA')

        if not os.path.isdir(target + '/' + p + '/INPUT'):
            os.mkdir(target + '/' + p + '/INPUT')

        if not os.path.isdir(target + '/' + p + '/OUTPUT'):
            os.mkdir(target + '/' + p + '/OUTPUT')

        # Get and Generate Required Files
        create_file_set(p)

        data = open(target + '/' + p + '/DATA/' + p + '_init.xyz', 'r').read()

        beads = int(re.search('nbead +(\d+)\n', data).group(1))

        output_str = p + ',' + str(beads) + ',' + target + '/' + p + '\n'

        with open(target + '/pull_pdb_output.log', 'a') as f:
            f.write(output_str)

        # Create Input Files
        for i in ['RL', 'thrust', 'CL']:
            print('\tCreating ' + i + ' Input File')
            output_str = '''set dynamics underdamped;
set temp 0.60;
set run 2718;
set restart off;
set istep_restart 0.0;
set nstep 10000;
set nup 10000;
set zeta 5e-2;
set boxl 750.0;
set cutofftype neighborlist;
set usegpu_nl 1;
set nl_algorithm ''' + i + ''';
set usegpu_pl 1;
set pl_algorithm ''' + i + ''';
set usegpu_vdw_energy 1;
set usegpu_vdw_force 1;
set usegpu_fene_energy 1;
set usegpu_fene_force 1;
set usegpu_ss_ang_energy 1;
set usegpu_ss_ang_force 1;
set usegpu_pos 1;
set usegpu_vel 1;
set usegpu_rand_force 1;
set usegpu_clear_force 1;
set nnlup 50;
set ncell 75.0; lcell = boxl / ncell = ~9.09
;
load bonds DATA/''' + p + '''_bonds.dat;
load angles DATA/''' + p + '''_angles.dat;
load vdw DATA/''' + p + '''_vdw.dat;
load init DATA/''' + p + '''_init.xyz;
;
set ufname OUTPUT/''' + p + '''.0.60.2718.out;
set cfname OUTPUT/''' + p + '''.0.60.2718.coords.out;
set unccfname OUTPUT/''' + p + '''.0.60.2718.coords_uncorrected.out;
set uncbinfname OUTPUT/''' + p + '''.0.60.2718.traj_uncorrected.bin;
set vfname OUTPUT/''' + p + '''.0.60.2718.velocs.out;
set binfname OUTPUT/''' + p + '''.0.60.2718.traj.bin;
set iccnfigfname OUTPUT/''' + p + '''.0.60.2718.iccnfig.xyz;
set rgenfname OUTPUT/''' + p + '''.0.60.2718.rgen.dat;
;
run;
    '''

            with open(target + '/' + p + '/INPUT/input_nl_' + i, 'w') as f:
                f.write(output_str)


        print('\tCreating CPU Input File')

        output_str = '''set dynamics underdamped;
set temp 0.60;
set run 2718;
set restart off;
set istep_restart 0.0;
set nstep 10000;
set nup 10000;
set zeta 5e-2;
set boxl 750.0;
set cutofftype neighborlist;
set usegpu_nl 0;
set usegpu_pl 0;
set usegpu_vdw_energy 0;
set usegpu_vdw_force 0;
set usegpu_fene_energy 0;
set usegpu_fene_force 0;
set usegpu_ss_ang_energy 0;
set usegpu_ss_ang_force 0;
set usegpu_pos 0;
set usegpu_vel 0;
set usegpu_rand_force 0;
set usegpu_clear_force 0;
set nnlup 50;
set ncell 75.0; lcell = boxl / ncell = ~9.09
;
load bonds DATA/''' + p + '''_bonds.dat;
load angles DATA/''' + p + '''_angles.dat;
load vdw DATA/''' + p + '''_vdw.dat;
load init DATA/''' + p + '''_init.xyz;
;
set ufname OUTPUT/''' + p + '''.0.60.2718.out;
set cfname OUTPUT/''' + p + '''.0.60.2718.coords.out;
set unccfname OUTPUT/''' + p + '''.0.60.2718.coords_uncorrected.out;
set uncbinfname OUTPUT/''' + p + '''.0.60.2718.traj_uncorrected.bin;
set vfname OUTPUT/''' + p + '''.0.60.2718.velocs.out;
set binfname OUTPUT/''' + p + '''.0.60.2718.traj.bin;
set iccnfigfname OUTPUT/''' + p + '''.0.60.2718.iccnfig.xyz;
set rgenfname OUTPUT/''' + p + '''.0.60.2718.rgen.dat;
;
run;
'''
        with open(target + '/' + p + '/INPUT/input_nl', 'w') as f:
            f.write(output_str)
            
    except KeyboardInterrupt:
        # quit
        sys.exit()

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        tb = traceback.extract_tb(exc_tb)[-1]
        print('An exception occurred with ' + p)
        print(e)
        print(exc_type, tb[2], tb[1])

        with open(target + '/pull_pdb_output.log', 'a') as f:
            f.write(p + ',NaN,' + target + '/' + p + '\n')
        
        if os.path.isdir(target + '/' + p):
            if os.path.isdir(target + '/' + p):
                shutil.rmtree(target + '/' + p)
        

def modify_pdb(pdb_file_contents):
    match = re.search('REMARK \d+?\s*?BEST REPRESENTATIVE CONFORMER IN THIS ENSEMBLE\s*?:\s*?(\d+)', pdb_file_contents)
    if not match == None:
        best_model = int(match.group(1))
        
        match_str = '(MODEL\s+' + str(best_model) + '\s+\n(ATOM\s+\d+\s+\w+\s+\w+\s+\w+\s+\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+\d+\.\d+\s+\w+\s+\n)+TER\s+\d+\s+\w+\s+\w+\s+\d+\s+\nENDMDL)'
        match = re.search(match_str, pdb_file_contents)
        model = match.group(1)
        '''
        match_str = '(HEADER(.+\n)+)MODEL\s+1'
        match = re.search(match_str, pdb_file_contents)
        header = match.group(1)
        '''
        full_output = model
        return full_output
    else:
        match = re.search('REMARK \d+?\s*?BEST REPRESENTATIVE CONFORMER IN THIS ENSEMBLE\s*?:\s*?(NULL)', pdb_file_contents)
        if not match == None:
            best_model = 1
        
            match_str = '(MODEL\s+' + str(best_model) + '\s+\n(ATOM\s+\d+\s+\w+\s+\w+\s+\w+\s+\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+\d+\.\d+\s+\w+\s+\n)+TER\s+\d+\s+\w+\s+\w+\s+\d+\s+\nENDMDL)'
            match = re.search(match_str, pdb_file_contents)
            model = match.group(1)
            '''
            match_str = '(HEADER(.+\n)+)MODEL\s+1'
            match = re.search(match_str, pdb_file_contents)
            header = match.group(1)
            '''
            full_output = model
            return full_output
        else:
            return pdb_file_contents


if __name__ == '__main__':
    random.seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--directory", dest="directory", help="Directory of folders")
    parser.add_argument("-i", "--input", dest="input", help="File containing PDBIDs to pull")
    parser.add_argument("-r", "--restart", dest="restart", default=0, help="Restart position", type=int)

    args = parser.parse_args()

    target = args.directory

    # proteins = pypdb.get_all()

    proteins = pypdb.Query('covid').search()
    #proteins = pypdb.Query('2MM4').search()
    sample = proteins
    #sample_size = int(sys.argv[2])

    sample = [x.upper().replace(' ', '') for x in open(args.input).read().splitlines()]

    #sample = random.sample(proteins, sample_size)
    output_str = 'Name,Nbead,Directory\n'

    if args.restart == 0:
        with open(target + '/pull_pdb_output.log', 'w+') as f:
            f.write(output_str)
            
    for p in sample[args.restart:]:
        pull_and_generate(p, target)
        