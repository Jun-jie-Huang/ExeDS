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


#echo "########################"
#echo "evaluate Dev set"
#echo ${MODEL_DIR}
#export SAVE_RESULT=${MODEL_DIR}/result_dev/
#echo ${SAVE_RESULT}
#mkdir ${SAVE_RESULT}
#CUDA_VISIBLE_DEVICES="0" python -u predict-gpt-neo.py \
#  --model_name_or_path ${MODEL_DIR} \
#  --test_file ${DATA_DIR}/dev.jsonl \
#  --max_input 2048 \
#  --write_generation_path ${SAVE_RESULT}/split_generation_results.json \
#  2>&1 |tee ${MODEL_DIR}/logs/gptneo_finetune_dev.log
#echo ${MODEL_DIR}
#echo ${SAVE_RESULT}
#mkdir ${SAVE_RESULT}/post_process
#python post_process.py \
#    --path_generation ${SAVE_RESULT}/split_generation_results.json \
#    --to_path ${SAVE_RESULT}/post_process/split_generation_results.json
#python ./CodeBLEU/evaluate.py \
#    --generation_dir ${SAVE_RESULT}/post_process  \
#    2>&1 |tee ${MODEL_DIR}/logs/evaluate_finetune_dev.log


pwd
echo "########################"
echo "evaluate Test set"
echo ${MODEL_DIR}
export SAVE_RESULT=${MODEL_DIR}/result_test/
echo ${SAVE_RESULT}
mkdir ${SAVE_RESULT}
CUDA_VISIBLE_DEVICES="0" python -u predict-gpt-neo.py \
  --model_name_or_path ${MODEL_DIR} \
  --test_file ${DATA_DIR}/test.jsonl \
  --max_input 2048 \
  --write_generation_path ${SAVE_RESULT}/split_generation_results.json \
  2>&1 |tee ${MODEL_DIR}/logs/gptneo_finetune_test.log
echo ${MODEL_DIR}
echo ${SAVE_RESULT}
mkdir ${SAVE_RESULT}/post_process
python post_process.py \
    --path_generation ${SAVE_RESULT}/split_generation_results.json \
    --to_path ${SAVE_RESULT}/post_process/split_generation_results.json
python ./CodeBLEU/evaluate.py \
    --generation_dir ${SAVE_RESULT}/post_process  \
    2>&1 |tee ${MODEL_DIR}/logs/evaluate_finetune_test.log


