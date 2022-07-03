#python train.py --dataset cifar10 --net cxfy42 --batch_size 128 --warmup 2 --num_epoch 60 --exp_id test_lip_cert --optimizer Adam --lr 0.01 --train_mode cert
#python train.py --dataset cifar10 --net cxfy42 --batch_size 128 --warmup 2 --num_epoch 60 --exp_id test_lip_nor --optimizer Adam --lr 0.01 --train_mode normal
#python train.py --dataset cifar10 --net cxfy42 --batch_size 128 --warmup 2 --num_epoch 60 --exp_id test_lip_adv --optimizer Adam --lr 0.01 --train_mode adv
#a="aa"
#echo $a
#python train.py --ord inf --optimizer SGD --lr 0.05 --dataset cifar10 --net cxfy42 --train_mode cert --exp_id 000 --batch_norm 1 --num_epoch 60 --warmup 1
#python train.py --ord inf --optimizer SGD --lr 0.05 --dataset cifar10 --net cxfy42 --train_mode normal --exp_id 000 --batch_norm 1 --num_epoch 60 --warmup 1
python train.py --dataset cifar10 --net cxfy42 --batch_size 128 --warmup 2 --num_epoch 200 --train_mode cert --optimizer SGD --lr 0.1 --exp_id robust_exp2 --batch_norm 1 --activation LeakyReLU

