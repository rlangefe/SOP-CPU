import pypdb
import random
import os
import sys
import subprocess

def create_file_set(p):
    print('Pulling ' + p)

    # Get PDB File
    pdb_file = pypdb.get_pdb_file(p)

    # Save PDB File
    f = open(target + '/' + p + '/DATA/' + p + '.pdb', 'w')
    f.write(pdb_file)
    print('\tWriting ' + p + ' PDB File')
    f.close()

    # Calculate and Save Bonds File
    f = open(target + '/' + p + '/DATA/' + p + '_bonds.dat', 'w+')
    contents = subprocess.check_output('perl gen_sop_input.pl -i ' + target + '/' + p + '/DATA/' + p + '.pdb -b', shell=True, stderr=subprocess.STDOUT).decode('utf-8')
    f.write(contents)
    print('\tWriting ' + p + ' Bonds File')
    f.close()

    # Calculate and Save Angles File
    f = open(target + '/' + p + '/DATA/' + p + '_angles.dat', 'w+')
    contents = subprocess.check_output('perl gen_sop_input.pl -i ' + target + '/' + p + '/DATA/' + p + '.pdb -a', shell=True, stderr=subprocess.STDOUT).decode('utf-8')
    f.write(contents)
    print('\tWriting ' + p + ' Angles File')
    f.close()

    # Calculate and Save VDW File
    f = open(target + '/' + p + '/DATA/' + p + '_vdw.dat', 'w+')
    contents = subprocess.check_output('perl gen_sop_input.pl -i ' + target + '/' + p + '/DATA/' + p + '.pdb -v', shell=True, stderr=subprocess.STDOUT).decode('utf-8')
    f.write(contents)
    print('\tWriting ' + p + ' VDW File')
    f.close()

    # Calculate and Save Init File
    f = open(target + '/' + p + '/DATA/' + p + '_init.xyz', 'w+')
    contents = subprocess.check_output('perl gen_sop_input.pl -i ' + target + '/' + p + '/DATA/' + p + '.pdb -x', shell=True, stderr=subprocess.STDOUT).decode('utf-8')
    f.write(contents)
    print('\tWriting ' + p + ' Init File')
    f.close()

def pull_and_generate(p):
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

        # Create Input Files
        for i in ['RL', 'thrust', 'CL']:
            print('\tCreating ' + i + ' Input File')
            str = '''set dynamics underdamped;
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
set usegpu_pos 0;
set usegpu_vel 1;
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
                f.write(str)


        print('\tCreating CPU Input File')

        str = '''set dynamics underdamped;
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
            f.write(str)
    except:
        print('An error occurred with ' + p)



if __name__ == '__main__':
    target = sys.argv[1]

    random.seed(42)

    # proteins = pypdb.get_all()

    proteins = pypdb.Query('covid').search()
    sample = proteins
    #sample_size = int(sys.argv[2])

    #sample = random.sample(proteins, sample_size)
    str = ''
    for p in sample:
        str = str + p + '\n'

    with open(target + '/pull_pdb_output.log', 'w+') as f:
        f.write(str)
            
    for p in sample:
        pull_and_generate(p)
        