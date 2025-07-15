#!/usr/bin/env -S bash --noprofile --norc
#PBS -A ParaLLMs
#PBS -N mixed-coords
#PBS -q prod
#PBS -l select=10
#PBS -l filesystems=home:eagle
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /home/shourya01/stormer_deepspeed/exp23.log

# Load required modules (modify as needed for your environment)
module use /soft/modulefiles
module load conda
conda activate base

# Load environment variables that support NCCL
export OMP_NUM_THREADS=4
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1
export TRITON_DISABLE_AUTOTUNE=1
export MPICH_GPU_SUPPORT_ENABLED=1

# Define the home dir and go there
PROJECT_DIR=$HOME/stormer_deepspeed
cd $PROJECT_DIR

# Copy host directory
cat $PBS_NODEFILE | uniq > $PROJECT_DIR/hostfile
HOSTFILE=$PROJECT_DIR/hostfile

# Define number of nodes and number of CPUs
NUM_NODES=$(cat $HOSTFILE | wc -l)
echo "Running on $NUM_NODES nodes"
GPUS_PER_NODE=4
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
echo "Total number of GPUs: $TOTAL_GPUS"

# Copy host info into DeepSpeed hostfile
> $PROJECT_DIR/ds_hostfile
while read -r node; do
    echo "$node slots=$GPUS_PER_NODE" >> $PROJECT_DIR/ds_hostfile
done < $HOSTFILE

# Run mpiexec
mpiexec --verbose --envall -n ${TOTAL_GPUS} -ppn ${GPUS_PER_NODE} --hostfile="${PBS_NODEFILE}"\
    python train_coords.py --deepspeed --tuning --lr_candidates 5e-4,1e-4,5e-5,1e-5,5e-6,1e-6,5e-7 --lora_rank 4 \
    --dataset_name mixed --use_weather --train_epochs 20 --experiment_name exp23_mixed_timesfm_stormer
# Done
echo "Training completed"