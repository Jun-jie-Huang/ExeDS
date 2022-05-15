export BASIC_DIR=../
BASIC_DIR=$1
MODEL=$2
PREPRO_DIR=$3
EPOCH=$4
PER_NODE_GPU=$5



DEV_GPU="0"
if [ ${PER_NODE_GPU} = 8 ]; then
  GPU="0,1,2,3,4,5,6,7"
elif [ ${PER_NODE_GPU} = 16 ]; then
  GPU="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
fi
echo ${GPU}
echo ${PER_NODE_GPU}


export DATA_DIR=${BASIC_DIR}/preprocessed_data/${PREPRO_DIR}/gpt_neo/
#export MODEL=EleutherAI/gpt-neo-125M
#export MODEL=EleutherAI/gpt-neo-1.3B
#export MODEL=EleutherAI/gpt-neo-2.7B
#export MODEL=flax-community/gpt-neo-125M-code-clippy

if [ "${MODEL}" = "EleutherAI/gpt-neo-125M" ]; then
    export BLOCK_SIZE=2048
    export BATCH_SIZE=3
    export MODEL_DIR=${BASIC_DIR}/saved_models/gpt_neo125/${PREPRO_DIR}_epoch${EPOCH}_block${BLOCK_SIZE}_bz${BATCH_SIZE}_G${PER_NODE_GPU}/
elif [ "${MODEL}" = "flax-community/gpt-neo-125M-code-clippy" ]; then
    export BLOCK_SIZE=2048
    export BATCH_SIZE=3
    export MODEL_DIR=${BASIC_DIR}/saved_models/gpt_neo_code_clippy/${PREPRO_DIR}_epoch${EPOCH}_block${BLOCK_SIZE}_bz${BATCH_SIZE}_G${PER_NODE_GPU}/
elif [ "${MODEL}" = "EleutherAI/gpt-neo-1.3B" ]; then
    export BLOCK_SIZE=2048
    export BATCH_SIZE=1
    export MODEL_DIR=${BASIC_DIR}/saved_models/gpt_neo13/${PREPRO_DIR}_epoch${EPOCH}_block${BLOCK_SIZE}_bz${BATCH_SIZE}_G${PER_NODE_GPU}/
elif [ "${MODEL}" = "EleutherAI/gpt-neo-2.7B" ]; then
    export BLOCK_SIZE=1536
    export BATCH_SIZE=1
    export MODEL_DIR=${BASIC_DIR}/saved_models/gpt_neo27/${PREPRO_DIR}_epoch${EPOCH}_block${BLOCK_SIZE}_bz${BATCH_SIZE}_G${PER_NODE_GPU}/
else
    echo "set a wrong parameter for --PRETRAIN_DIR"
    exit 1
fi
mkdir ${MODEL_DIR}
mkdir ${MODEL_DIR}/logs
echo "setting MODEL=$MODEL, BLOCK_SIZE=$BLOCK_SIZE, BATCH_SIZE=$BATCH_SIZE, PER_NODE_GPU=$PER_NODE_GPU"
pwd

python -m torch.distributed.launch \
    --nproc_per_node ${PER_NODE_GPU} finetune-gpt-neo.py \
    --model_name_or_path ${MODEL} \
    --validation_split_percentage 1 \
    --block_size ${BLOCK_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size $((BATCH_SIZE*2)) \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --do_train \
    --do_eval \
    --train_file ${DATA_DIR}/train.jsonl \
    --validation_file ${DATA_DIR}/dev.jsonl \
    --output_dir ${MODEL_DIR} \
    --overwrite_output_dir \
    --logging_steps 1000000 \
    --logging_dir ${MODEL_DIR}/logs \
    --save_steps 1000 \
    --num_train_epochs 10 \
    --deepspeed ds_config.json \
    --report_to tensorboard \
    --run_name "${PREPRO_DIR}-$(date +%b%d-%H-%M-%Y)" \
    --fp16 \
    2>&1 |tee ${MODEL_DIR}/logs/gptneo_train.log
    # --fp16_backend amp \
    # --fp16_opt_level O1

