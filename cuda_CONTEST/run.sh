#!/bin/bash

# Run vector sum;
#bin/b -d -c -n 100000000 -b vec -I 1 -i 30 -t 64;
#bin/b -d -c -n 100000000 -b vec -I 2 -i 30 -t 64;

# Run matrix multiplication;
#bin/b -d -c -n 1000 -b mmul -I 1 -i 30 -t 8;
#bin/b -d -c -n 1000 -b mmul -I 2 -i 30 -t 8 -B 14;

# Run PPR;
#bin/b -d -c -n 1000 -b ppr -I 0 -i 1 -t 64;
bin/b -d -b ppr -i 1 -I 2 -s 0 -t 4 -g './data/small.mtx';
#bin/b -d -b ppr -I 2 -s 0 -t 256 -i 1 -g './data/California.mtx';
#bin/b -d -b ppr -I 1 -s 0 -t 76 -g './data/wikipedia.mtx';
