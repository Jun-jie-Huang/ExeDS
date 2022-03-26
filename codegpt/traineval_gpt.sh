export BASIC_DIR=../
BASIC_DIR=$1
PRETRAIN_DIR=$2
PREPRO_DIR=$3
EPOCH=$4
PER_NODE_GPU=$5


DEV_GPU="0"
if [ ${PER_NODE_GPU} = 8 ]; then
  GPU="0,1,2,3,4,5,6,7"
elif [ ${PER_NODE_GPU} = 16 ]; then
  GPU="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
fi
echo $GPU
if [ "${PER_NODE_GPU}" = 8 ]; then
    export TRAIN_BATCH=6
else
    export TRAIN_BATCH=16
fi


export DATA_DIR=${BASIC_DIR}/preprocessed_data/${PREPRO_DIR}
#export PRETRAIN_DIR=microsoft/CodeGPT-small-py-adaptedGPT2
#export PRETRAIN_DIR=microsoft/CodeGPT-small-py
#export PRETRAIN_DIR=microsoft/gpt2

if [ "${PRETRAIN_DIR}" = "microsoft/CodeGPT-small-py-adaptedGPT2" ]; then
    export MODEL_DIR=${BASIC_DIR}/saved_models/codegpt_adapt/codegpt_adapt_${PREPRO_DIR}_epoch${EPOCH}_G${PER_NODE_GPU}/
elif [ "${PRETRAIN_DIR}" = "microsoft/CodeGPT-small-py" ]; then
    export MODEL_DIR=${BASIC_DIR}/saved_models/codegpt/codegpt_${PREPRO_DIR}_epoch${EPOCH}_G${PER_NODE_GPU}/
elif [ "${PRETRAIN_DIR}" = "gpt2" ]; then
    export MODEL_DIR=${BASIC_DIR}/saved_models/gpt2/gpt2_${PREPRO_DIR}_epoch${EPOCH}_G${PER_NODE_GPU}/
else
    echo "set a wrong parameter for --PRETRAIN_DIR"
    exit 1
fi
pwd
mkdir -p ${MODEL_DIR}
mkdir -p ${MODEL_DIR}/logs


python  -m torch.distributed.launch --nproc_per_node=${PER_NODE_GPU} run.py \
        --data_dir ${DATA_DIR} \
        --langs py \
        --output_dir ${MODEL_DIR} \
        --pretrain_dir ${PRETRAIN_DIR} \
        --model_type gpt2 \
        --block_size 512 \
        --do_train \
        --evaluate_during_training \
        --node_index 0 \
        --gpu_per_node ${PER_NODE_GPU} \
        --learning_rate 5e-5 \
        --weight_decay 0.01 \
        --per_gpu_train_batch_size ${TRAIN_BATCH} \
        --per_gpu_eval_batch_size 25 \
        --warmup_steps 1000 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs ${EPOCH} \
        --logging_steps 500 \
        --save_steps 3000 \
        --overwrite_output_dir \
        --seed 42 \
        --dev_file_type dev \
        2>&1 |tee ${MODEL_DIR}/logs/codegpt_train.log


echo "##################################################################"
export PRETRAIN_DIR=${MODEL_DIR}/checkpoint-last
CUDA_VISIBLE_DEVICES=${DEV_GPU} python -u run.py \
        --data_dir ${DATA_DIR} \
        --langs py \
        --output_dir ${MODEL_DIR} \
        --pretrain_dir ${PRETRAIN_DIR} \
        --model_type gpt2 \
        --block_size 512 \
        --do_eval \
        --logging_steps 400 \
        --seed 42 \
        2>&1 |tee ${MODEL_DIR}/logs/codegpt_dev.log

echo "##################################################################"
echo "EVAL_CHECKPOINT to eval the checkpoint: ${EVAL_CHECKPOINT}"
echo ${MODEL_DIR}
echo "evaluate $2 Dev set  (surface form)"
python ./CodeBLEU/evaluate.py \
        --gold_file ${MODEL_DIR}/dev.gold \
        --generation_file ${MODEL_DIR}/dev.output \
        --save_score_path ${MODEL_DIR}/surface_form_score_dev.json \
        2>&1 |tee ${MODEL_DIR}/logs/evaluate_dev.log


export PRETRAIN_DIR=${MODEL_DIR}/checkpoint-last
CUDA_VISIBLE_DEVICES=${DEV_GPU} python -u run.py \
        --data_dir ${DATA_DIR} \
        --langs py \
        --output_dir ${MODEL_DIR} \
        --pretrain_dir ${PRETRAIN_DIR} \
        --model_type gpt2 \
        --block_size 512 \
        --do_infer \
        --logging_steps 400 \
        --seed 42 \
        2>&1 |tee ${MODEL_DIR}/logs/codegpt_test.log

echo "##################################################################"
echo "EVAL_CHECKPOINT to eval the checkpoint: ${EVAL_CHECKPOINT}"
echo ${MODEL_DIR}
echo "evaluate $2 Test set (surface form)"
python ./CodeBLEU/evaluate.py \
        --gold_file ${MODEL_DIR}/test.gold \
        --generation_file ${MODEL_DIR}/test.output \
        --save_score_path ${MODEL_DIR}/surface_form_score_test.json \
        2>&1 |tee ${MODEL_DIR}/logs/evaluate_test.log

