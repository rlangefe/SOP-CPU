#!/usr/bin/perl -w

# Description: Convert a SOP trajectory into PDB MODEL format
# them out in PDB format.

use strict;
use Getopt::Std;

# make sure command line argments okay
if ($#ARGV == -1) {
    &usage() and exit -1;
}

# defaults and global variables
my $traj;
my $numbeads;

# evaluate any flags that may change the default values
&getopts('i:n:', \my %opts);

# SOP Model trajectory
if (defined $opts{'i'}) {
    $traj = $opts{'i'};
} else {
    &usage() and exit;
}

# number of beads
if (defined $opts{'n'}) {
    $numbeads = $opts{'n'};
} else {
    &usage() and exit;
}

# open SOP model trajectory file to read in structures
open(TRAJ, "$traj")
    or die "Unable to open $traj.";

my $title = <TRAJ>; # read in and ignore title
my $iframe = 1; # each frame
my $iatom = 0; # each bead
my @beadinfo = ();

# calculate the intermolecular contacts in each frame
while (my $cline = <TRAJ>) {

    $iatom++;
    (my $x, my $y, my $z) = split (/\s+/, $cline);
    push(@beadinfo,"$iatom:$x:$y:$z");

    if ($iatom == $numbeads) {
	&printframe(\@beadinfo, $iframe);
	$iatom = 0;
	$iframe++;
	# initialize list for next frame
	@beadinfo = ();
    }

}

exit 0;

sub printframe {
    my $coorlistref = shift;
    my $coorlist = @$coorlistref;
    my $iframe = shift;

    my $icoor = 0;
    printf("MODEL %8d\n", $iframe);
    foreach my $resi (@beadinfo) {
	(my $at, my $x, my $y, my $z) = split(":", $resi);
	printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
	       "ATOM", $at, "CA ", "GLY", "A", $at, $x, $y, $z, 1.00, 0.00) if ($at < 10000);
    }
    print "TER\nENDMDL\n";
}


sub usage {
    print STDERR << "EOF";

usage: $0 -i SOP_trajectory_file -n number_of_beads

 -i       : SOP trajectory file
 -n       : number of beads

example: $0 -i 1ehz.traj -n 192

EOF
}

