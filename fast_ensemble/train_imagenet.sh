NAME=eps2

CONFIG1=imagenet/configs_phase1.yml
CONFIG2=imagenet/configs_phase2.yml
CONFIG3=imagenet/configs_phase3.yml

PREFIX1=fast_adv_phase1_${NAME}
PREFIX2=fast_adv_phase2_${NAME}
PREFIX3=fast_adv_phase3_${NAME}

OUT1=fast_train_phase1_${NAME}.out
OUT2=fast_train_phase2_${NAME}.out
OUT3=fast_train_phase3_${NAME}.out

EVAL1=fast_eval_phase1_${NAME}.out
EVAL2=fast_eval_phase2_${NAME}.out
EVAL3=fast_eval_phase3_${NAME}.out

END1=fast_adv_phase1_${NAME}_step2_eps2_repeat1/checkpoint_epoch6.pth.tar
END2=fast_adv_phase2_${NAME}_step2_eps2_repeat1/checkpoint_epoch12.pth.tar
END3=trained_models/fast_adv_phase3_${NAME}_step2_eps2_repeat1/checkpoint_epoch15.pth.tar

# training for phase 1
python -u train_imagenet.py -c $CONFIG1 --output_prefix $PREFIX1 | tee $OUT1

# evaluation for phase 1
# python -u main_fast.py $DATA160 -c $CONFIG1 --output_prefix $PREFIX1 --resume $END1  --evaluate | tee $EVAL1

# training for phase 2
python -u train_imagenet.py -c $CONFIG2 --output_prefix $PREFIX2 --resume $END1 | tee $OUT2

# evaluation for phase 2
# python -u main_fast.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END2 --evaluate | tee $EVAL2

# training for phase 3
python -u train_imagenet.py -c $CONFIG3 --output_prefix $PREFIX3 --resume $END2 | tee $OUT3

# evaluation for phase 3
# python -u main_fast.py $DATA -c $CONFIG3 --output_prefix $PREFIX3 --resume $END3 --evaluate | tee $EVAL3
