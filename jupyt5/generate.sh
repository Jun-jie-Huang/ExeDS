export BASIC_DIR=../
BASIC_DIR=$1
MODEL=$2
PREPRO_DIR=$3
MAX_TOKENS=3600

export DATA_DIR=${BASIC_DIR}/preprocessed_data/${PREPRO_DIR}/fairseq_tokenization/normal/
if [ "${MODEL}" = "pymt5" ]; then
  export FINETUNE_FROM=${BASIC_DIR}/jupyt5_weights/checkpoint_best.pt
  export MODEL_DIR=${BASIC_DIR}/saved_models/pymt5/${PREPRO_DIR}_maxtokens${MAX_TOKENS}_G16/
else
  export FINETUNE_FROM=${BASIC_DIR}/jupyt5_weights/checkpoint_5.pt
  export MODEL_DIR=${BASIC_DIR}/saved_models/jupyt5/${PREPRO_DIR}_maxtokens${MAX_TOKENS}_G16/
fi
export EVAL_CHECKPOINT=${MODEL_DIR}/checkpoint_best.pt
mkdir ${MODEL_DIR}
mkdir ${MODEL_DIR}/logs

echo "####################################################"
echo "training DATA_DIR is: ${DATA_DIR}"
ls ${DATA_DIR}
echo "training data PREPRO_DIR is from : ${PREPRO_DIR};"
echo "FINETUNE_FROM the checkpoint in: ${FINETUNE_FROM}"
echo "MODEL_DIR to save model: ${MODEL_DIR}"
echo "EVAL_CHECKPOINT to eval the checkpoint: ${EVAL_CHECKPOINT}"
echo "##############"

#export CUDA_LAUNCH_BLOCKING=1
echo "EVAL_CHECKPOINT to eval the checkpoint: ${EVAL_CHECKPOINT}"
export SAVE_RESULT=${MODEL_DIR}/result_valid/
echo "save the results to: ${SAVE_RESULT}"
echo ${DATA_DIR}
fairseq-generate ${DATA_DIR} \
    -s "src" -t "tgt" \
    --gen-subset valid \
    --path ${EVAL_CHECKPOINT} \
    --log-format simple \
    --results-path ${SAVE_RESULT} \
    --beam 5 \
    --remove-bpe \
    --skip-invalid-size-inputs-valid-test \
    2>&1 |tee ${MODEL_DIR}/logs/fairseq_dev.log

echo "##############"
echo "EVAL_CHECKPOINT to eval the checkpoint: ${EVAL_CHECKPOINT}"
export SAVE_RESULT=${MODEL_DIR}/result_test/
echo ${SAVE_RESULT}
echo ${DATA_DIR}
fairseq-generate ${DATA_DIR} \
    -s "src" -t "tgt" \
    --gen-subset test \
    --path ${EVAL_CHECKPOINT} \
    --log-format simple \
    --results-path ${SAVE_RESULT} \
    --beam 5 \
    --remove-bpe \
    --skip-invalid-size-inputs-valid-test \
    2>&1 |tee ${MODEL_DIR}/logs/fairseq_test.log


echo "###########"
echo "evaluate Dev set"
echo ${MODEL_DIR}
export SAVE_RESULT=${MODEL_DIR}/result_valid/
echo ${SAVE_RESULT}
python ./get_splited_generation_results.py \
    --fairseq_generate_result_path ${SAVE_RESULT}/generate-valid.txt \
    --write_path ${SAVE_RESULT}/split_generation_results.json
python ./CodeBLEU/evaluate.py \
    --generation_dir ${SAVE_RESULT}  \
    2>&1 |tee ${MODEL_DIR}/logs/evaluate_dev.log

echo "###########"
echo "evaluate Test set"
echo ${MODEL_DIR}
export SAVE_RESULT=${MODEL_DIR}/result_test/
echo ${SAVE_RESULT}
python ./get_splited_generation_results.py \
    --fairseq_generate_result_path ${SAVE_RESULT}/generate-test.txt \
    --write_path ${SAVE_RESULT}/split_generation_results.json
python ./CodeBLEU/evaluate.py \
    --generation_dir ${SAVE_RESULT}  \
    2>&1 |tee ${MODEL_DIR}/logs/evaluate_test.log



