#python train.py --dataset cifar10 --net cxfy42 --batch_size 128 --warmup 2 --num_epoch 60 --exp_id test_lip_cert --optimizer Adam --lr 0.01 --train_mode cert
#python train.py --dataset cifar10 --net cxfy42 --batch_size 128 --warmup 2 --num_epoch 60 --exp_id test_lip_nor --optimizer Adam --lr 0.01 --train_mode normal
#python train.py --dataset cifar10 --net cxfy42 --batch_size 128 --warmup 2 --num_epoch 60 --exp_id test_lip_adv --optimizer Adam --lr 0.01 --train_mode adv
#a="aa"
#echo $a
#python train.py --ord inf --optimizer SGD --lr 0.05 --dataset cifar10 --net cxfy42 --train_mode cert --exp_id 000 --batch_norm 1 --num_epoch 60 --warmup 1
#python train.py --ord inf --optimizer SGD --lr 0.05 --dataset cifar10 --net cxfy42 --train_mode normal --exp_id 000 --batch_norm 1 --num_epoch 60 --warmup 1
python train.py --dataset cifar10 --net cxfy42 --batch_size 128 --warmup 2 --num_epoch 200 --train_mode cert --optimizer SGD --lr 0.1 --exp_id robust_exp2 --batch_norm 1 --activation LeakyReLU

python train.py --net cxfy42 --train_mode cert --exp_id 002 --num_epoch 10 --warmup 0

python train.py --net cxfy42 --train_mode cert --exp_id 002 --eta_dn 0.5


opt=SGD
act=GeLU
id=0
for batchSize in 128 256 512
do
  for opt in SGD Adam
  do
    for act in ReLU LeakyReLU GeLU
    do
      for id in 0 1 2
      do
        python train.py --net vgg16 --train_mode normal --val_mode normal  --optimizer ${opt} --act ${act} --batch_size $batchSize --dir 10run/${act}_${opt}_${batchSize} --exp_id 0${id}
        python train.py --net vgg16 --train_mode cert --eta_dn 0.25 --dn_rate 0.95 --balance 1 --val_mode normal  --optimizer ${opt} --act ${act} --batch_size ${batchSize} --dir 10run/${act}_${opt}_${batchSize} --exp_id gen_1${id}
        python train.py --net vgg16 --train_mode cert --eta_dn 0.1 --dn_rate 0.95 --balance 0 --val_mode normal  --optimizer ${opt} --act ${act} --batch_size ${batchSize} --dir 10run/${act}_${opt}_${batchSize} --exp_id gen_0${id}
      done
    done
done
done


for batchSize in 128 256
do
  for act in LeakyReLU ReLU
  do
    for id in 0 1 2 3 4
    do
      qsub pytorch-detection.sh --net cxfy42 --train_mode std --act ${act} --batch_size ${batchSize} --dir 10run_cifar10/${act}_${batchSize} --exp_id std_0${id}
      qsub pytorch-detection.sh --net cxfy42 --train_mode cer --act ${act} --batch_size ${batchSize} --dir 10run_cifar10/${act}_${batchSize} --exp_id gen_0${id} --eta_dn 0.1 --dn_rate 0.9 --balance 0
    done
  done
done


act=LeakyReLU
python train.py --net vgg16 --train_mode std --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_01 --exp_id bm --noise_sigma 0.1 --cert_input noise
python train.py --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_01 --exp_id 00 --noise_sigma 0.1 --cert_input noise --eta_float 0.0 --float_loss 0.01
python train.py --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_01 --exp_id 01 --noise_sigma 0.1 --cert_input noise --eta_float 0.1 --float_loss 0.01
python train.py --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_01 --exp_id 02 --noise_sigma 0.1 --cert_input noise --eta_float 0.2 --float_loss 0.01
python train.py --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_01 --exp_id 10 --noise_sigma 0.1 --cert_input noise --eta_float 0.0 --float_loss 0.01
python train.py --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_01 --exp_id 11 --noise_sigma 0.1 --cert_input noise --eta_float 0.1 --float_loss 0.01
python train.py --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_01 --exp_id 12 --noise_sigma 0.1 --cert_input noise --eta_float 0.2 --float_loss 0.01




for act in LeakyReLU
do
	qsub pytorch-detection.sh --net vgg16 --train_mode std --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id std
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id bnm --noise_sigma 0.1 --cert_input noise
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id nof --noise_sigma 0.1 --cert_input noise --float_loss 0.05 --eta_float 0.0
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id lip --noise_sigma 0.1 --cert_input noise --eta_float 0.0  --lip 1
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id adv --noise_sigma 0.1 --cert_input images --noise_type FGSM

	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 01 --noise_sigma 0.1 --cert_input noise --float_loss 0.05 --eta_float 0.1
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 02 --noise_sigma 0.1 --cert_input noise --float_loss 0.05 --eta_fixed 0.05
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 03 --noise_sigma 0.1 --cert_input noise --float_loss 0.05 --eta_float 0.1 --eta_fixed 0.05


	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 11 --noise_sigma 0.1 --cert_input noise --float_loss 0.05 --eta_float 0.1 --lip 1
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 12 --noise_sigma 0.1 --cert_input noise --float_loss 0.05 --eta_fixed 0.05 --lip 1
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 13 --noise_sigma 0.1 --cert_input noise --float_loss 0.05 --eta_float 0.1 --eta_fixed 0.05 --lip 1

	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 21 --noise_sigma 0.1 --cert_input noise --float_loss 0.05 --eta_float 0.1 --balance 1
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 22 --noise_sigma 0.1 --cert_input noise --float_loss 0.05 --eta_fixed 0.05 --balance 1
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 23 --noise_sigma 0.1 --cert_input noise --float_loss 0.05 --eta_float 0.1 --eta_fixed 0.05 --balance 1

	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 31 --noise_sigma 0.1 --cert_input images --float_loss 0.05 --eta_float 0.1 --balance 1
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 32 --noise_sigma 0.1 --cert_input images --float_loss 0.05 --eta_fixed 0.05 --balance 1
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 33 --noise_sigma 0.1 --cert_input images --float_loss 0.05 --eta_float 0.1 ---eta_fixed 0.05 --balance 1

	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 41 --noise_sigma 0.1 --cert_input noise --float_loss 0.05 --eta_float 0.1 --balance 1 --noise_type FGSM
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 42 --noise_sigma 0.1 --cert_input noise --float_loss 0.05 --eta_fixed 0.05 --balance 1 --noise_type FGSM
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 43 --noise_sigma 0.1 --cert_input noise --float_loss 0.05 --eta_float 0.1 --eta_fixed 0.05 --balance 1 --noise_type FGSM

	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 51 --noise_sigma 0.1 --cert_input images --float_loss 0.05 --eta_float 0.1 --balance 1 --noise_type FGSM
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 52 --noise_sigma 0.1 --cert_input images --float_loss 0.05 --eta_fixed 0.05 --balance 1 --noise_type FGSM
	qsub pytorch-detection.sh --net vgg16 --train_mode cer --val_mode adv --act ${act} --batch_size $batchSize --dir robust/benchmark_02 --exp_id 53 --noise_sigma 0.1 --cert_input images --float_loss 0.05 --eta_float 0.1 --eta_fixed 0.05 --balance 1 --noise_type FGSM
done


python train.py --net cxfy42 --val_mode adv --act LeakyReLU --batch_size 128 --dir robust/fgsm_test --lr_scheduler cyclic --num_epoch 15 --warmup 0 --train_mode cer --exp_id cer_0 --attack FGSM --noise_type FGSM --eta_float -0.2 --lip_layers 1 --lip_loss 0.01
python train.py --net cxfy42 --val_mode adv --act LeakyReLU --batch_size 128 --dir robust/fgsm_test --lr_scheduler cyclic --num_epoch 15 --warmup 0 --train_mode cer --exp_id cer_0 --attack FGSM --noise_type FGSM --eta_fixed -0.1 --lip_layers 1 --lip_loss 0.05 --lip_inverse 1
python train.py --net cxfy42 --val_mode adv --act LeakyReLU --batch_size 128 --dir robust/fgsm_test --lr_scheduler cyclic --num_epoch 15 --warmup 0 --train_mode adv --exp_id adv_0