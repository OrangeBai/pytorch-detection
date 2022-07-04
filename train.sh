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
      qsub pytorch-detection.sh --net cxfy42 --train_mode std --act ${act} --batch_size ${batchSize} --dir 10run/${act}_${batchSize} --exp_id std_0${id}
      qsub pytorch-detection.sh --net cxfy42 --train_mode cer --act ${act} --batch_size ${batchSize} --dir 10run/${act}_${batchSize} --exp_id gen_0${id} --eta_dn 0.1 --dn_rate 0.9 --balance 0
    done
  done
done