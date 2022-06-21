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

if [ "$lmd" = "0.2" ]; then
    echo "Strings are equal."
else
    echo "Strings are not equal."
fi

echo s_${lmd}
