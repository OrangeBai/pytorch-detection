lmd=0.01
bnd=0.01
eta=0.8
net=vgg16
dataset=cifar100

while getopts "l:b:n:d:e:" opt; do
  case $opt in
  l) lmd=$OPTARG ;;
  b) bnd=$OPTARG ;;
  n) net=$OPTARG ;;
  d) dataset=$OPTARG ;;
  e) eta=$OPTARG;;
  esac
done

echo $net
echo $dataset
echo $lmd
echo $bnd
echo $eta
echo l_${lmd}_b_${bnd}_e_${eta}
python train.py --net $net --dataset $dataset --exp_id l_${lmd}_b_${bnd}_eta_${eta} --lmd $lmd --bnd $bnd --eta $eta
