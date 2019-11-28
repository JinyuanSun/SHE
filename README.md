# SHE(one-letter code for Serine, Histidine and Glutamic acid)
Redesign protein to a Ser-His-Glu hydrolase  
Usage:
-  demo: `python3 SHE_v3_1.py demo`, make sure you have all the files downloaded in a same directory and installed dssp higher than 3.0.  
-  serial: `python3 SHE_v3_1.py $your_pdb`, this will take a while depend on your protein. In this version, the GLY and PRO will not be take into account (PRO are important to the turn and GLY is highly filxable). For a 2 GHz processor, the A chian of 3wzl will take a week.
-  multiple threads: `python3 SHE_v3_1.py $your_pdb nt`, It's well known that python do not really support multiple threads. This multiple threads version is actually divide a big list into many small list and run this code individually. The 3wzl_A has been tested, the whole cauculation took 15 hours on a 56 threads 2 GHz work station.

This is based on  
-  [backboe dependent library](http://dunbrack.fccc.edu/bbdep2010/Tutorial.php), the 3 simple.lib files are derived from the 2010bbdep.lib. In this stricpt, the chi2 of HIS and chi3 of glu are not connsidered as non-rotamtic, in the further version, it may be improved.  
-  [PeptideBuilder package](https://github.com/mtien/PeptideBuilder), you don't need to install it, but some lines in the code are inspried by that package.  
-  [dssp](https://swift.cmbi.umcn.nl/gv/dssp/DSSP_3.html), a good software with long history, SHE use dssp to calculate psi and phi of the backbone.
