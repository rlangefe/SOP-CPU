#!/usr/bin/perl
# read in a PDB file and output SOP input files

use strict;
use Getopt::Std;
use Math::Trig;

# make sure command line argments okay
if ($#ARGV == -1) {
    &usage() and exit -1;
}

# defaults and global variables
my $dconv = 90.00/&acos(0.00);
my $dconv = 1.0;

my $coor_file;
my $numres;
my $segid;
my $frame;
my $printPDB;
my $printbonds;
my $printangles;
my $printvdw;
my $printnc;
my $printstruct;

# evaluate any flags that may change the default values
&getopts('i:bavnxp', \my %opts);

# coordinate file
if (defined $opts{'i'}) {
    $coor_file = $opts{'i'};
} else {
    &usage() and exit;
}

# write out coordinates of SOP in PDB format
if (defined $opts{'p'}) {
    $printPDB = 1;
}

# write out bonds
if (defined $opts{'b'}) {
    $printbonds = 1;
}

# write out angles
if (defined $opts{'a'}) {
    $printangles = 1;
}

# write out vdw
if (defined $opts{'v'}) {
    $printvdw = 1;
}

# write out native contact list
if (defined $opts{'n'}) {
    $printnc = 1;
}

# write out initial structure
if (defined $opts{'x'}) {
    $printstruct = 1;
}

# open coordinate file
open(COOR, "$coor_file")
    or die "Unable to open $coor_file.";

my @xavg = ();
my @yavg = ();
my @zavg = ();
my @numatom = ();
my @segid = ();
my @resname = ();
my @resnum = ();

my $ires = 0;
my $currres = 0;
my $prevres = 0;

while (my $line = <COOR>) {
    my $id = substr($line, 0, 6); $id =~ s/\s+//g;
    next unless ($id eq "ATOM");

    my $rnum = substr($line, 22, 6); $rnum =~ s/\s+//g;

    $currres = $rnum;
    if ($currres != $prevres) {
	$prevres = $currres;
	$ires++;
    }

    my $atmname = substr($line, 11, 5); $atmname =~ s/\s+//g;
    $resnum[$ires] = $rnum;
    $resname[$ires] = substr($line, 17, 4); $resname[$ires] =~ s/\s+//g;
    $segid[$ires] = substr($line, 21, 1); $segid[$ires] =~ s/\s+//g;

    # coordinates
    my $x = substr($line, 30, 8); $x =~ s/\s+//g;
    my $y = substr($line, 38, 8); $y =~ s/\s+//g;
    my $z = substr($line, 46, 8); $z =~ s/\s+//g;

    if (&isheavy($atmname)) {
	$xavg[$ires] += $x;
	$yavg[$ires] += $y;
	$zavg[$ires] += $z;
	$numatom[$ires]++;
    }
}

# get average coordinates for each bead
for (my $i=1; $i <= $ires; $i++) {

    if ($numatom[$i] > 0) {
	$xavg[$i] = $xavg[$i] / $numatom[$i];
	$yavg[$i] = $yavg[$i] / $numatom[$i];
	$zavg[$i] = $zavg[$i] / $numatom[$i];
    } else {
	die "No coordinates found for residue $i";
    }

}

if (defined $printbonds) {

    my @bondlist = ();
    my $nbnd = 0;
    for (my $i=1; $i <= $ires - 1; $i++) {

	if ($segid[$i] eq $segid[$i+1]) {
	    push (@bondlist, sprintf("%d %d %.6f\n", $i, $i+1,
				     &calc_dist($xavg[$i],$yavg[$i],$zavg[$i],
						$xavg[$i+1],$yavg[$i+1],$zavg[$i+1])));
	    $nbnd++;
	}
    }

    printf("nbnd %d\n", $nbnd);
    foreach my $line (@bondlist) {
	(my $i, my $j, my $dist) = split (/\s+/, $line);
	printf("%d %d %.6f\n", $i, $j, $dist);
    }
}

if (defined $printangles) {

    my @anglist = ();
    my $nang = 0;
    for (my $i=1; $i <= $ires - 2; $i++) {

	if ($segid[$i] eq $segid[$i+2]) {
	    push (@anglist, sprintf("%d %d %d %.6f\n", $i, $i+1, $i+2,
				    &calc_dist($xavg[$i],$yavg[$i],$zavg[$i],
					       $xavg[$i+2],$yavg[$i+2],$zavg[$i+2])));
	    $nang++;
	}
    }

    printf("nangle %d\n", $nang);
    foreach my $line (@anglist) {
	(my $i, my $j, my $k, my $ang) = split (/\s+/, $line);
	printf("%d %d %d %.6f\n", $i, $j, $k, $ang);
    }
}

if (defined $printvdw) {

    # add up number of native contacts and others
    my $natt;
    my $nrep;
    my @nblist = ();

    my $itype = 0;
    my $jtype = 0;
    my $rcut;

    for (my $i=1; $i <= $ires - 3; $i++) {
	for (my $j=$i + 3; $j <= $ires; $j++) {

	    my $dist = &calc_dist($xavg[$i],$yavg[$i],$zavg[$i],
				  $xavg[$j],$yavg[$j],$zavg[$j]);

	    if (&isprot($resname[$i]) != 0) {
		$itype = 1;
	    } elsif (&isnucl($resname[$i]) != 0) {
		$itype = 2;
	    } else {
		die "residue/nucleotide type unknown: $resname[$i]";
	    }

	    if (&isprot($resname[$j]) != 0) {
		$jtype = 1;
	    } elsif (&isnucl($resname[$j]) != 0) {
		$jtype = 2;
	    } else {
		die "residue/nucleotide type unknown: $resname[$j]";
	    }

            if ($itype == 2 && $jtype == 2) { # RNA-RNA
		$rcut = 14.0;
	    } elsif ($itype == 1 && $jtype == 1) { # prot-prot
		$rcut = 8.0;
	    } elsif (($itype == 1 && $jtype == 2) || ($itype == 2 && $jtype == 1)) { # prot-RNA
		$rcut = (14.0 + 8.0) / 2; # = 11
	    }


	    if ($dist < $rcut) {
		$natt++;
		push (@nblist, sprintf("%d %d %.6f %s %s\n", $i, $j, $dist, $itype, $jtype));
	    } else {
		$nrep++;
		push (@nblist, sprintf("%d %d %.6f %s %s\n", $i, $j, $dist, $itype, $jtype));
	    }
	}
    }

    print "natt $natt nrep $nrep epsilon_h 0.7 epsilon_l 1.0 sigma 7.0 rcut_rna 14.0 rcut_prot 8.0\n";

    foreach my $line (@nblist) {
	(my $i, my $j, my $dist, my $itype, my $jtype) = split (/\s+/, $line);
	printf("%d %d %.6f %d %d\n", $i, $j, $dist, $itype, $jtype);
    }
}

if (defined $printnc) {

    my $itype = 0;
    my $jtype = 0;
    my $rcut;

    for (my $i=1; $i <= $ires - 3; $i++) {
	for (my $j=$i+3; $j <= $ires; $j++) {

	    my $dist = &calc_dist($xavg[$i],$yavg[$i],$zavg[$i],
				  $xavg[$j],$yavg[$j],$zavg[$j]);

	    if (&isprot($resname[$i]) != 0) {
		$itype = 1;
	    } elsif (&isnucl($resname[$i]) != 0) {
		$itype = 2;
	    } else {
		die "residue/nucleotide type unknown: $resname[$i]";
	    }

	    if (&isprot($resname[$j]) != 0) {
		$jtype = 1;
	    } elsif (&isnucl($resname[$j]) != 0) {
		$jtype = 2;
	    } else {
		die "residue/nucleotide type unknown: $resname[$j]";
	    }

	    # type 1 = protein
	    # type 2 = nucl
            if ($itype == 2 && $jtype == 2) { # RNA-RNA
		$rcut = 14.0;
	    } elsif ($itype == 1 && $jtype == 1) { # prot-prot
		$rcut = 8.0;
	    } elsif (($itype == 1 && $jtype == 2) || ($itype == 2 && $jtype == 1)) { # prot-RNA
		$rcut = (14.0 + 8.0) / 2; # = 11
	    }


	    if ($dist < $rcut) {
		printf("%d %d %.6f %d %d\n", $i, $j, $dist, $itype, $jtype) if ($i < $j);
	    }
	}
    }
}


if (defined $printstruct) {
    my $type = 0;

    printf("nbead %d\n", $ires);
    for (my $i=1; $i <= $ires; $i++) {
	if (&isprot($resname[$i]) != 0) {
	    $type = 1;
	} elsif (&isnucl($resname[$i]) != 0) {
	    $type = 2;
	} else {
	    die "residue/nucleotide type unknown: $resname[$i]";
	}

	# type 1 = protein
	# type 2 = nucl

	printf("%d %.6f %.6f %.6f %d\n", $i, $xavg[$i],$yavg[$i],$zavg[$i], $type);
    }
}

if (defined $printPDB) {
    for (my $i=1; $i <= $ires; $i++) {
	printf("ATOM %6d  CA  %s %s%4d %11.3f%8.3f%8.3f  1.00  0.00\n", $i, $resname[$i], $segid[$i], $resnum[$i], $xavg[$i],$yavg[$i],$zavg[$i]);
    }
}


exit 0;

sub calc_dist {
    my $ax = shift;
    my $ay = shift;
    my $az = shift;

    my $bx = shift;
    my $by = shift;
    my $bz = shift;

    my $dist;

    my $mx = $bx - $ax;
    my $my = $by - $ay;
    my $mz = $bz - $az;

    $dist = sqrt($mx*$mx + $my*$my + $mz*$mz);

    return $dist;
}

sub calc_theta {
    my $ax = shift;
    my $ay = shift;
    my $az = shift;

    my $bx = shift;
    my $by = shift;
    my $bz = shift;

    my $cx = shift;
    my $cy = shift;
    my $cz = shift;

    my $costheta = ( ( ($ax - $bx) * ($cx - $bx) +
                       ($ay - $by) * ($cy - $by) +
                       ($az - $bz) * ($cz - $bz) )
                     / ( &calc_dist($ax, $ay, $az, $bx, $by, $bz)
                         * &calc_dist($bx, $by, $bz, $cx, $cy, $cz)));

    my $theta = &acos($costheta) * $dconv;

    return $theta
}

sub isprot {
    my $type = shift;

    if ($type eq  "ALA") {
	return (1);
    }
    elsif ($type eq  "ARG") {
	return (2);
    }
    elsif ($type eq  "ASN") {
	return (3);
    }
    elsif ($type eq  "ASP") {
	return (4);
    }
    elsif ($type eq  "CYS") {
	return (5);
    }
    elsif ($type eq  "GLN") {
	return (6);
    }
    elsif ($type eq  "GLU") {
	return (7);
    }
    elsif ($type eq  "GLY") {
	return (8);
    }
    elsif ($type eq  "HIS" or $type eq "HSD") {
	return (9);
    }
    elsif ($type eq  "ILE") {
	return (10);
    }
    elsif ($type eq  "LEU") {
	return (11);
    }
    elsif ($type eq  "LYS") {
	return (12);
    }
    elsif ($type eq  "MET") {
	return (13);
    }
    elsif ($type eq  "PHE") {
	return (14);
    }
    elsif ($type eq  "PRO") {
	return (15);
    }
    elsif ($type eq  "SER") {
	return (16);
    }
    elsif ($type eq  "THR") {
	return (17);
    }
    elsif ($type eq  "TRP") {
	return (18);
    }
    elsif ($type eq  "TYR") {
	return (19);
    }
    elsif ($type eq  "VAL") {
	return (20);
    }
    else {
	return(0); # otherwise unknown
    }
}

sub isnucl {
    my $type = shift;

    if ($type eq  "GUA" or $type eq "G") {
	return (1);
    }
    elsif ($type eq  "ADE" or $type eq "A") {
	return (2);
    }
    elsif ($type eq  "CYT" or $type eq "C") {
	return (3);
    }
    elsif ($type eq  "URA" or $type eq "U") {
	return (4);
    } else {
	return(0); # otherwise unknown
    }
}

sub isheavy {
    my $atmname = shift;

    if ($atmname !~ /^H/) {
	return 1;
    } else { 
	return 0;
    }

}

sub usage {
    print STDERR << "EOF";

usage: $0 -i PDB_file [-b][-a][-v][-x]

 -i       : input PDB structure file
 -b       : print out bonds file
 -a       : print out angles file
 -v       : print out VDW native contacts file
 -x       : print out initial structure file

example: $0 -i 1eor.pdb -b

EOF
}
