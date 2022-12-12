read -p "Enter dataset_model: " dataset_model
#M1:dataset_name = 'mnist' model_name = 'CNN_Mnist'
#C1:dataset_name = 'cifar10' model_name = 'ResNet18_Cifar10'
#W1:dataset_name = 'wikitext2' model_name = 'transformer'
#E1:dataset_name = 'emnist' model_name = 'VGG11_EMNIST'
read -p "Enter is_iid: " is_iid
#T: True F: False

master='localhost'
workers='localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost'
echo 'master(coordinator): '$master
echo 'worker_hosts: '$workers
world_size=0
for i in $workers
do
		world_size=$((world_size+1))
done

home='/YOURPATH/GWPF'
read -p "Specify allocated GPU-ID (world_size: $world_size): " cuda 
trial_no=$(ls $home/Logs/ | wc -l)
log_dir=$home/Logs/${trial_no}_${dataset_model}_${is_iid}

sudo mkdir -p $log_dir/code_snapshot_${trial_no} 
sudo mkdir -p $log_dir/params
cp $home/train_10n.sh $log_dir/code_snapshot_${trial_no}
cp $home/*.py $log_dir/code_snapshot_${trial_no}
sudo rm -f Latest_Log && ln -s $log_dir Latest_Log
echo 'logs in '$log_dir

num=0
number=5
startcuda=$cuda
echo 'number'$number
echo 'startcuda'$startcuda
job=worker_process.py

for i in $workers
do
	command="python $home/$job --master_address=tcp://${master}:$((20000+trial_no)) --rank=$num --world_size=$world_size --trial_no=$trial_no --dataset_model=$dataset_model --is_iid=$is_iid"
	echo $command
	CUDA_VISIBLE_DEVICES=$cuda nohup $command > $log_dir/worker_$num.log 2>&1 &
    echo 'cuda' $cuda
	num=$((num+1))
	cuda=$((startcuda+num%number))
done
