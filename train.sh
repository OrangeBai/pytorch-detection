python train.py --dataset cifar10 --net vgg16 --batch_size 128 --warmup 2 --num_epoch 60 --exp_id test_lip --optimizer Adam --lr 0.001 --train_mode cert
python train.py --dataset cifar10 --net vgg16 --batch_size 128 --warmup 2 --num_epoch 60 --exp_id test_lip --optimizer Adam --lr 0.001 --train_mode normal
python train.py --dataset cifar10 --net vgg16 --batch_size 128 --warmup 2 --num_epoch 60 --exp_id test_lip --optimizer Adam --lr 0.001 --train_mode adv