#!/bin/bash
#SBATCH --job-name=nemo_rl_sft_baselinemix_luciole
#SBATCH --output=logs/log_%x-%j.stdout
#SBATCH --error=logs/log_%x-%j.stdout
#SBATCH --gres=gpu:4
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --constraint h100
#SBATCH --account wuh@h100
#SBATCH --qos qos_gpu_h100-t3
#SBATCH --mail-type=ARRAY_TASKS,BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-type=ARRAY_TASKS,BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=achaturvedi@linagora.com,cthompson@linagora.com

#set -euo pipefail

## load environment
module purge
module load arch/h100
module load cuda/12.8.0
module load cudnn/9.10.2.21-12-cuda
module load nccl/2.27.3-1-cuda
module load uv/0.8.3
source .venv/bin/activate

export TORCH_CUDA_ARCH_LIST=9.0

export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_NUM_GPUS=$SLURM_GPUS_ON_NODE
export RAY_NUM_CPUS=$SLURM_CPUS_ON_NODE

export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# InfiniBand setup

IB_IFACE=$(ip -4 -o addr show | awk '{print $2}' | grep -E '^(ib|hsn|sl)' | head -n 1)

if [ -z "$IB_IFACE" ]; then
  echo "ERROR: No InfiniBand interface with IPv4 found"
  ip -4 addr show
  exit 1
fi

IB_IP_CMD="ip -4 addr show $IB_IFACE | awk '/inet / {split(\$2,a,\"/\"); print a[1]}'"

# NCCL setup

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=$IB_IFACE
export NCCL_ASYNC_ERROR_HANDLING=1


# nodes

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "$IB_IP_CMD")

if [ -z "$head_node_ip" ]; then
  echo "ERROR: Failed to resolve head node IB IP"
  exit 1
fi

export RAY_HEAD_IP="$head_node_ip"
export RAY_ADDRESS="$RAY_HEAD_IP:6379"

echo "Ray head node: $head_node ($RAY_HEAD_IP) via $IB_IFACE"

# Ray cluster

srun --nodes=$SLURM_JOB_NUM_NODES \
     --ntasks=$SLURM_JOB_NUM_NODES \
     --ntasks-per-node=1 \
     bash -c "
set -euo pipefail
local_ip=\$($IB_IP_CMD)
# IMPORTANT: Each node must identify as its OWN local IB IP
export RAY_NODE_IP_ADDRESS=\$local_ip

echo \"[\$(hostname)] IFACE=$IB_IFACE IP=\$local_ip CPUs=$SLURM_CPUS_PER_TASK GPUs=4\"

if [ \"\$SLURM_PROCID\" -eq 0 ]; then
    ray start --head \
      --node-ip-address=\$local_ip \
      --port=6379 \
      --num-cpus=$SLURM_CPUS_PER_TASK \
      --num-gpus=4 \
      --disable-usage-stats \
      --block
else
    # Wait for the head node to be reachable
    until (echo > /dev/tcp/$RAY_HEAD_IP/6379) >/dev/null 2>&1; do
      echo \"Waiting for Ray head at $RAY_HEAD_IP...\"
      sleep 2
    done
    ray start \
      --address=$RAY_ADDRESS \
      --node-ip-address=\$local_ip \
      --num-cpus=$SLURM_CPUS_PER_TASK \
      --num-gpus=4 \
      --disable-usage-stats \
      --block
fi
" &

# Safety wait

sleep 60
ray status || { echo "Ray failed to start"; exit 1; }

# run code

uv run --no-build-isolation python examples/run_sft.py \
  --config sft_8B_baseline_luciole.yaml \
  cluster.num_nodes=8 \
  cluster.gpus_per_node=4

