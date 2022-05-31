#$ -S /bin/bash

source activate pytorch


VAR=""
for ARG in "$@";do
	VAR+="${ARG} "
done
echo $VAR

python train.py $VAR
