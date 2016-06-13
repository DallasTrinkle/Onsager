#!/usr/bin/perl
use warnings;
use strict;

# The first line of trans (the tag) is ignored; the command line specifies the CONTCAT:
# ../trans.pl trans ../$(head -n 1 trans)/CONTCAR

scalar @ARGV == 2 or die "Needs exactly two arguments: trans CONTCAR\n";
open TRANS, $ARGV[0] or die "Cannot open transformation file ($ARGV[0]), $!";
open CONTCAR, $ARGV[1] or die "Cannot open CONTCAR file ($ARGV[1]), $!";

my (@trans, @gmat, @disp, @mapping, @pos, $i);
# Read the entire transformation file, pull out the pieces (ignore the first line):
@trans = <TRANS>;
@gmat = ( [ split(" ", $trans[1]) ], [ split(" ", $trans[2]) ], [ split(" ", $trans[3]) ] );
@disp = split(" ", $trans[4]);
@mapping = split(" ", $trans[5]);

sub grot {
    my @pos = @_;
    my (@u, $i, $j);
    for $i (0..2) {
	push @u, $gmat[$i][0]*$pos[0] + $gmat[$i][1]*$pos[1] + $gmat[$i][2]*$pos[2] + $disp[$i];
	if ($u[$i] >= 1.0) { $u[$i] -= 1.0; }
	if ($u[$i] <  0.0) {$u[$i] += 1.0;}
    }
    printf " %19.16lf %19.16lf %19.16lf\n", @u;
}

# Read the CONTCAR file; modify the first line, output read of header:
$_ = <CONTCAR>;
print "Transformed $_";
while (<CONTCAR>) {
    print;
    (/^D/ or /^d/) and last;
}
# get all the positions, and then print them in permuted order
while (<CONTCAR>) { push @pos, [ split ]; }
for $i (@mapping) { grot((@{$pos[$i]}, @gmat, @disp)); }