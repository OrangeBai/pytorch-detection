#python train.py --dataset cifar10 --net cxfy42 --batch_size 128 --warmup 2 --num_epoch 60 --exp_id test_lip_cert --optimizer Adam --lr 0.01 --train_mode cert
#python train.py --dataset cifar10 --net cxfy42 --batch_size 128 --warmup 2 --num_epoch 60 --exp_id test_lip_nor --optimizer Adam --lr 0.01 --train_mode normal
#python train.py --dataset cifar10 --net cxfy42 --batch_size 128 --warmup 2 --num_epoch 60 --exp_id test_lip_adv --optimizer Adam --lr 0.01 --train_mode adv
a="aa"
echo $a