python3 preprocess.py --split train --file_name exeds_train.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 3 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split dev --file_name exeds_dev.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 3 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split test --file_name exeds_test.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 3 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900

MODEL_DICT="../jupyt5_weights/dict.src.txt"
DATADIR="../preprocessed_data/prepro_addTab-df_madeup_token_range3_lineLen1-25_c200m200a900/fairseq_tokenization"
fairseq-preprocess -s "src" -t "tgt" --srcdict ${MODEL_DICT} --joined-dictionary --destdir ${DATADIR}/normal --trainpref "${DATADIR}/python.train_nl_to_code" --validpref "${DATADIR}/python.dev_nl_to_code" --testpref "${DATADIR}/python.test_nl_to_code" --workers 24


##### context_range=5
python3 preprocess.py --split train --file_name exeds_train.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 5 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split dev --file_name exeds_dev.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 5 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split test --file_name exeds_test.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 5 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900

MODEL_DICT="../jupyt5_weights/dict.src.txt"
DATADIR="../preprocessed_data/prepro_addTab-df_madeup_token_range5_lineLen1-25_c200m200a900/fairseq_tokenization"
fairseq-preprocess -s "src" -t "tgt" --srcdict ${MODEL_DICT} --joined-dictionary --destdir ${DATADIR}/normal --trainpref "${DATADIR}/python.train_nl_to_code" --validpref "${DATADIR}/python.dev_nl_to_code" --testpref "${DATADIR}/python.test_nl_to_code" --workers 24


##### context_range=4
python3 preprocess.py --split train --file_name exeds_train.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 4 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split dev --file_name exeds_dev.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 4 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split test --file_name exeds_test.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 4 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900

MODEL_DICT="../jupyt5_weights/dict.src.txt"
DATADIR="../preprocessed_data/prepro_addTab-df_madeup_token_range4_lineLen1-25_c200m200a900/fairseq_tokenization"
fairseq-preprocess -s "src" -t "tgt" --srcdict ${MODEL_DICT} --joined-dictionary --destdir ${DATADIR}/normal --trainpref "${DATADIR}/python.train_nl_to_code" --validpref "${DATADIR}/python.dev_nl_to_code" --testpref "${DATADIR}/python.test_nl_to_code" --workers 24


##### context_range=2
python3 preprocess.py --split train --file_name exeds_train.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 2 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split dev --file_name exeds_dev.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 2 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split test --file_name exeds_test.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 2 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900

MODEL_DICT="../jupyt5_weights/dict.src.txt"
DATADIR="../preprocessed_data/prepro_addTab-df_madeup_token_range2_lineLen1-25_c200m200a900/fairseq_tokenization"
fairseq-preprocess -s "src" -t "tgt" --srcdict ${MODEL_DICT} --joined-dictionary --destdir ${DATADIR}/normal --trainpref "${DATADIR}/python.train_nl_to_code" --validpref "${DATADIR}/python.dev_nl_to_code" --testpref "${DATADIR}/python.test_nl_to_code" --workers 24


##### context_range=1
python3 preprocess.py --split train --file_name exeds_train.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 1 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split dev --file_name exeds_dev.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 1 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split test --file_name exeds_test.json --do_fairseq_tokenization --do_gptneo --token_type token --context_range 1 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900

MODEL_DICT="../jupyt5_weights/dict.src.txt"
DATADIR="../preprocessed_data/prepro_addTab-df_madeup_token_range1_lineLen1-25_c200m200a900/fairseq_tokenization"
fairseq-preprocess -s "src" -t "tgt" --srcdict ${MODEL_DICT} --joined-dictionary --destdir ${DATADIR}/normal --trainpref "${DATADIR}/python.train_nl_to_code" --validpref "${DATADIR}/python.dev_nl_to_code" --testpref "${DATADIR}/python.test_nl_to_code" --workers 24


##### token_type=str
python3 preprocess.py --split train --file_name exeds_train.json --do_fairseq_tokenization --do_gptneo --token_type str --context_range 3 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split dev --file_name exeds_dev.json --do_fairseq_tokenization --do_gptneo --token_type str --context_range 3 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split test --file_name exeds_test.json --do_fairseq_tokenization --do_gptneo --token_type str --context_range 3 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900

MODEL_DICT="../jupyt5_weights/dict.src.txt"
DATADIR="../preprocessed_data/prepro_addTab-df_madeup_str_range3_lineLen1-25_c200m200a900/fairseq_tokenization"
fairseq-preprocess -s "src" -t "tgt" --srcdict ${MODEL_DICT} --joined-dictionary --destdir ${DATADIR}/normal --trainpref "${DATADIR}/python.train_nl_to_code" --validpref "${DATADIR}/python.dev_nl_to_code" --testpref "${DATADIR}/python.test_nl_to_code" --workers 24


#### no add_table, token_type=token
python3 preprocess.py --split train --file_name exeds_train.json --do_fairseq_tokenization --do_gptneo --not_add_table --token_type token --context_range 3 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split dev --file_name exeds_dev.json --do_fairseq_tokenization --do_gptneo --not_add_table --token_type token --context_range 3 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900
python3 preprocess.py --split test --file_name exeds_test.json --do_fairseq_tokenization --do_gptneo --not_add_table --token_type token --context_range 3 --max_code_cell_tokens 200 --max_md_cell_tokens 200 --max_ctx_cell_tokens 900

MODEL_DICT="../jupyt5_weights/dict.src.txt"
DATADIR="../preprocessed_data/prepro_noTab_madeup_token_range3_lineLen1-25_c200m200a900/fairseq_tokenization"
fairseq-preprocess -s "src" -t "tgt" --srcdict ${MODEL_DICT} --joined-dictionary --destdir ${DATADIR}/normal --trainpref "${DATADIR}/python.train_nl_to_code" --validpref "${DATADIR}/python.dev_nl_to_code" --testpref "${DATADIR}/python.test_nl_to_code" --workers 24





