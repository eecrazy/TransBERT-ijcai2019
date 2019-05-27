
  CUDA_VISIBLE_DEVICES=$(free-gpu) python run_rocstories_1.5.py \
  --bert_model /home/zyli/github/copa/examples/mnli_output/model2 \
  --do_lower_case \
  --seed 17763 \
  --margin 0.31 \
  --do_train \
  --do_eval \
  --do_test \
  --do_prediction 0 \
  --data_dir /export/b01/zyli/data/ROCStories \
  --train_batch_size 20 \
  --do_margin_loss 0 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --max_seq_length 128 \
  --output_dir roc_output
done



CUDA_VISIBLE_DEVICES=$(free-gpu) python run_rocstories_1.5.py \
  --bert_model /home/zyli/github/copa/examples/mnli_output_large/model1 \
  --seed 17763 \
  --margin 0.31 \
  --do_train \
  --do_eval \
  --do_test \
  --do_prediction 1 \
  --data_dir /export/b01/zyli/data/ROCStories \
  --train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --eval_batch_size 8 \
  --do_margin_loss 0 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --max_seq_length 70 \
  --output_dir roc15_output





#验证 en nc ec 有没有帮助

CUDA_VISIBLE_DEVICES=$(free-gpu) python run_rocstories.py \
  --bert_model /home/zyli/github/copa/examples/mnli_output/model2 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.28 \
  --l2_reg 0.02 \
  --do_train \
  --do_eval \
  --do_test \
  --do_prediction 0 \
  --data_dir /export/b01/zyli/data/ROCStories \
  --train_batch_size 20 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 100 \
  --do_margin_loss 0 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --max_seq_length 128 \
  --output_dir roc10_output

# baseline: bert-base-uncased and +MNLI
# 0.882950293960449 0.9 186 32436 0.28 0 1e-05 bert-base-uncased run_rocstories.py
*# 0.8808123997862106 0.9 172 32436 0.28 0 1e-05
*# 0.9064671298770711 0.94 257 32436 0.28 0 

CUDA_VISIBLE_DEVICES=$(free-gpu) python run_rocstories.py \
  --bert_model bert-large-cased \
  --seed 32436 \
  --margin 0.28 \
  --l2_reg 0.02 \
  --do_train \
  --do_eval \
  --do_test \
  --do_prediction 1 \
  --data_dir /export/b01/zyli/data/ROCStories \
  --train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --eval_batch_size 8 \
  --do_margin_loss 0 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --max_seq_length 70 \
  --output_dir roc_output/bert_large

# 在100开发集上调参: bert-large-cased and +MNLI
*# 0.9053981827899519 0.92 260 32436 0.28 0 1e-05 bert-large-cased

*# 0.9166221272047034 0.96     32436 0.28 0 1e-05 /home/zyli/github/copa/examples/mnli_output_large/model1  从第三轮开始dev
# * 0.9230358097274185 0.95 195 32436 0.28 0 1e-05 /home/zyli/github/copa/examples/mnli_output_large/model1 run_rocstories.py

# *# 0.9005879208979155 0.94 191 32436 0.28 0 1e-05 bert-large-cased run_rocstories.py  从第二轮开始dev
# 0.9160876536611438 0.96 171 32436 0.28 0 1e-05 /home/zyli/github/copa/examples/mnli_output_large/model1 run_rocstories.py   从第二轮开始dev







# /home/zyli/github/copa/examples/mnli_nc/model1
0.8877605558524853 0.93 187 32436 0.28 0 1e-05 /home/zyli/github/copa/examples/mnli_nc/model1 run_rocstories.py

# /home/zyli/github/copa/examples/mnli_ec/model2
0.892036344200962 0.94 151 32436 0.28 0 1e-05 /home/zyli/github/copa/examples/mnli_ec/model2 run_rocstories.py

# /home/zyli/github/copa/examples/mnli_en/model1
0.8615713522180652 0.89 100 32436 0.28 0 1e-05 /home/zyli/github/copa/examples/mnli_en/model1 run_rocstories.py

# /home/zyli/github/copa/examples/snli_output/model0
0.8476750400855158 0.89 90 32436 0.28 0 1e-05 /home/zyli/github/copa/examples/snli_output/model0 run_rocstories.py

# /home/zyli/github/copa/examples/snli_output/model2
0.8594334580438269 0.9 90 32436 0.28 0 1e-05 /home/zyli/github/copa/examples/snli_output/model2 run_rocstories.py

# /home/zyli/github/copa/examples/swag_output/model4
0.8845537145911277 0.94 134 32436 0.28 0 1e-05 /home/zyli/github/copa/examples/swag_output/model4 run_rocstories.py

# /home/zyli/github/copa/examples/swag_output/model2
0.8760021378941742 0.94 106 32436 0.28 0 1e-05 /home/zyli/github/copa/examples/swag_output/model2 run_rocstories.py

# /home/zyli/github/copa/examples/mnli_2choice_output/model2
0.8952431854623196 0.93 137 32436 0.28 0 1e-05 /home/zyli/github/copa/examples/mnli_2choice_output/model2 run_rocstories.py

# /home/zyli/github/copa/examples/msnli_output/model1
0.8893639764831641 0.92 172 32436 0.28 0 1e-05 /home/zyli/github/copa/examples/msnli_output/model1 run_rocstories.py

# /home/zyli/github/copa/examples/mnli_ec_no_bert/model2
0.5034740780331374 0.61 142 32436 0.28 0 1e-05 /home/zyli/github/copa/examples/mnli_ec_no_bert/model2 run_rocstories.py

# /home/zyli/github/copa/examples/mnli_output_no_bert/model2
0.4537680384820951 0.57 95 32436 0.28 0 1e-05 /home/zyli/github/copa/examples/mnli_output_no_bert/model2 run_rocstories.py



