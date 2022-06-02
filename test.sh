#$ -S /bin/bash

source activate pytorch


VAR=""
for i in 1 2 3 4 5; do
	echo "$1_$i"
done


#python train.py $VAR
