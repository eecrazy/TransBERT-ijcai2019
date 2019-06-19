CUDA_VISIBLE_DEVICES=$(free-gpu) python run_mnli_1.py \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --do_train \
  --do_eval \
  --do_test \
  --do_prediction 0 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new/1 \
  --train_batch_size 50 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 100 \
  --do_margin_loss 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output


26175
2722
2759
2740

# best_eval_acc1: 0.8280675973548861 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc2: 0.869155491119971 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc3: 0.9445255474452555 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py

# best_eval_acc1: 0.8306392358559882 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc2: 0.8811163465023559 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc3: 0.9445255474452555 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py

best_eval_acc1: 0.8335782512858193 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py
best_eval_acc2: 0.8832910474809713 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py
best_eval_acc3: 0.9448905109489051 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py

# best_eval_acc1: 0.8376193975018369 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc2: 0.8528452337803551 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc3: 0.9335766423357664 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py

# best_eval_acc1: 0.8368846436443791 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc2: 0.8814787966654585 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc3: 0.941970802919708 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py

best_eval_acc1: 0.8479059515062454 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py
best_eval_acc2: 0.8753171438927148 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py
best_eval_acc3: 0.9408759124087591 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py




CUDA_VISIBLE_DEVICES=$(free-gpu) python run_mnli_2.py \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --do_train \
  --do_eval \
  --do_test \
  --do_prediction 0 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new/2 \
  --train_batch_size 50 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 100 \
  --do_margin_loss 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output

17451
2722
2759
# best_eval_acc1: 0.8122703894195444 32436 0.2 1 2e-05 bert-base-uncased run_mnli_2.py
# best_eval_acc2: 0.8564697354113809 32436 0.2 1 2e-05 bert-base-uncased run_mnli_2.py

# best_eval_acc1: 0.8251285819250551 32436 0.2 1 2e-05 bert-base-uncased run_mnli_2.py
# best_eval_acc2: 0.8720550924247916 32436 0.2 1 2e-05 bert-base-uncased run_mnli_2.py

best_eval_acc1: 0.8302718589272594 32436 0.2 1 2e-05 bert-base-uncased run_mnli_2.py
best_eval_acc2: 0.8789416455237404 32436 0.2 1 2e-05 bert-base-uncased run_mnli_2.py

# best_eval_acc1: 0.8335782512858193 32436 0.2 0 2e-05 bert-base-uncased run_mnli_2.py
# best_eval_acc2: 0.8513954331279449 32436 0.2 0 2e-05 bert-base-uncased run_mnli_2.py

# best_eval_acc1: 0.8438648052902278 32436 0.2 0 2e-05 bert-base-uncased run_mnli_2.py
# best_eval_acc2: 0.8789416455237404 32436 0.2 0 2e-05 bert-base-uncased run_mnli_2.py

best_eval_acc1: 0.8464364437913299 32436 0.2 0 2e-05 bert-base-uncased run_mnli_2.py
best_eval_acc2: 0.8822036969916637 32436 0.2 0 2e-05 bert-base-uncased run_mnli_2.py



CUDA_VISIBLE_DEVICES=$(free-gpu) python run_mnli_3.py \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --do_train \
  --do_eval \
  --do_test \
  --do_prediction 0 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new/3 \
  --train_batch_size 50 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 100 \
  --do_margin_loss 0 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output


9189
4130
4132

best_eval_acc1: 0.7392251815980629 32436 0.2 1 2e-05 bert-base-uncased run_mnli_3.py
best_eval_acc2: 0.8216360116166506 32436 0.2 1 2e-05 bert-base-uncased run_mnli_3.py

# best_eval_acc1: 0.7460048426150121 32436 0.2 1 2e-05 bert-base-uncased run_mnli_3.py
# best_eval_acc2: 0.7879961277831559 32436 0.2 1 2e-05 bert-base-uncased run_mnli_3.py

# best_eval_acc1: 0.7392251815980629 32436 0.2 1 2e-05 bert-base-uncased run_mnli_3.py
# best_eval_acc2: 0.7904162633107454 32436 0.2 1 2e-05 bert-base-uncased run_mnli_3.py


best_eval_acc1: 0.7292978208232446 32436 0.2 0 2e-05 bert-base-uncased run_mnli_3.py
best_eval_acc2: 0.8051790900290416 32436 0.2 0 2e-05 bert-base-uncased run_mnli_3.py

# best_eval_acc1: 0.7358353510895884 32436 0.2 0 2e-05 bert-base-uncased run_mnli_3.py
# best_eval_acc2: 0.7729912875121007 32436 0.2 0 2e-05 bert-base-uncased run_mnli_3.py

# best_eval_acc1: 0.737772397094431 32436 0.2 0 2e-05 bert-base-uncased run_mnli_3.py
# best_eval_acc2: 0.7628267182962246 32436 0.2 0 2e-05 bert-base-uncased run_mnli_3.py

# 密度图绘制

1:
CUDA_VISIBLE_DEVICES=$(free-gpu) python plot_mnli.py \
  --bert_model /export/b01/zyli/data/mnli_copa_2choice_new/1/model2_0 \
  --do_margin_loss 0 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new/1 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output


CUDA_VISIBLE_DEVICES=$(free-gpu) python plot_mnli.py \
  --bert_model /export/b01/zyli/data/mnli_copa_2choice_new/1/model2_1 \
  --do_margin_loss 1 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new/1 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output



2:
CUDA_VISIBLE_DEVICES=$(free-gpu) python plot_mnli.py \
  --bert_model /export/b01/zyli/data/mnli_copa_2choice_new/2/model2_0 \
  --do_margin_loss 0 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new/2 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output


CUDA_VISIBLE_DEVICES=$(free-gpu) python plot_mnli.py \
  --bert_model /export/b01/zyli/data/mnli_copa_2choice_new/2/model2_1 \
  --do_margin_loss 1 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new/2 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output



3:
CUDA_VISIBLE_DEVICES=$(free-gpu) python plot_mnli.py \
  --bert_model /export/b01/zyli/data/mnli_copa_2choice_new/3/model2_0 \
  --do_margin_loss 0 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new/3 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output


CUDA_VISIBLE_DEVICES=$(free-gpu) python plot_mnli.py \
  --bert_model /export/b01/zyli/data/mnli_copa_2choice_new/3/model2_1 \
  --do_margin_loss 1 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new/3 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output




















CUDA_VISIBLE_DEVICES=$(free-gpu) python run_joci_2choice_1.py \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --seed 30684 \
  --margin 0.2 \
  --do_train \
  --do_eval \
  --do_test \
  --do_prediction 0 \
  --data_dir /export/b01/zyli/data/joci_copa_2choice_new/1 \
  --train_batch_size 20 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --do_margin_loss 0 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir joci_2choice_output

8698
830
603
1567

best_eval_acc1: 0.7710843373493976 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_1.py
best_eval_acc2: 0.8308457711442786 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_1.py
best_eval_acc3: 0.9361837906828334 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_1.py

# best_eval_acc1: 0.7650602409638554 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_1.py
# best_eval_acc2: 0.8175787728026535 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_1.py
# best_eval_acc3: 0.9329929802169751 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_1.py

# best_eval_acc1: 0.7566265060240964 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_1.py
# best_eval_acc2: 0.8126036484245439 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_1.py
# best_eval_acc3: 0.9329929802169751 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_1.py

best_eval_acc1: 0.7578313253012048 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_1.py
best_eval_acc2: 0.8325041459369817 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_1.py
best_eval_acc3: 0.9355456285896617 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_1.py

# best_eval_acc1: 0.7626506024096386 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_1.py
# best_eval_acc2: 0.8208955223880597 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_1.py
# best_eval_acc3: 0.9285258455647735 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_1.py

# best_eval_acc1: 0.7626506024096386 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_1.py
# best_eval_acc2: 0.8109452736318408 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_1.py
# best_eval_acc3: 0.9234205488194002 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_1.py



CUDA_VISIBLE_DEVICES=$(free-gpu) python run_joci_2choice_2.py \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --seed 30684 \
  --margin 0.2 \
  --do_train \
  --do_eval \
  --do_test \
  --do_prediction 0 \
  --data_dir /export/b01/zyli/data/joci_copa_2choice_new/2 \
  --train_batch_size 20 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --do_margin_loss 0 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir joci_2choice_output


4159
830
603
best_eval_acc1: 0.7542168674698795 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_2.py
best_eval_acc2: 0.8225538971807629 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_2.py

# best_eval_acc1: 0.7650602409638554 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_2.py
# best_eval_acc2: 0.814262023217247 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_2.py

# best_eval_acc1: 0.7710843373493976 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_2.py
# best_eval_acc2: 0.8059701492537313 30684 0.2 1 1e-05 bert-base-uncased run_joci_2choice_2.py

best_eval_acc1: 0.763855421686747 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_2.py
best_eval_acc2: 0.835820895522388 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_2.py

# best_eval_acc1: 0.7807228915662651 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_2.py
# best_eval_acc2: 0.8175787728026535 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_2.py

# best_eval_acc1: 0.7795180722891566 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_2.py
# best_eval_acc2: 0.8159203980099502 30684 0.2 0 1e-05 bert-base-uncased run_joci_2choice_2.py

# 密度图绘制

1:
CUDA_VISIBLE_DEVICES=$(free-gpu) python plot_joci.py \
  --bert_model /export/b01/zyli/data/joci_copa_2choice_new/1/model2_0 \
  --do_margin_loss 0 \
  --data_dir /export/b01/zyli/data/joci_copa_2choice_new/1 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir joci_2choice_output


CUDA_VISIBLE_DEVICES=$(free-gpu) python plot_joci.py \
  --bert_model /export/b01/zyli/data/joci_copa_2choice_new/1/model2_1 \
  --do_margin_loss 1 \
  --data_dir /export/b01/zyli/data/joci_copa_2choice_new/1 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir joci_2choice_output



2:
CUDA_VISIBLE_DEVICES=$(free-gpu) python plot_joci.py \
  --bert_model /export/b01/zyli/data/joci_copa_2choice_new/2/model2_0 \
  --do_margin_loss 0 \
  --data_dir /export/b01/zyli/data/joci_copa_2choice_new/2 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir joci_2choice_output


CUDA_VISIBLE_DEVICES=$(free-gpu) python plot_joci.py \
  --bert_model /export/b01/zyli/data/joci_copa_2choice_new/2/model2_1 \
  --do_margin_loss 1 \
  --data_dir /export/b01/zyli/data/joci_copa_2choice_new/2 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir joci_2choice_output


3:
CUDA_VISIBLE_DEVICES=$(free-gpu) python plot_joci.py \
  --bert_model /export/b01/zyli/data/joci_copa_2choice_new/3/model2_0 \
  --do_margin_loss 0 \
  --data_dir /export/b01/zyli/data/joci_copa_2choice_new/3 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir joci_2choice_output


CUDA_VISIBLE_DEVICES=$(free-gpu) python plot_joci.py \
  --bert_model /export/b01/zyli/data/joci_copa_2choice_new/3/model2_1 \
  --do_margin_loss 1 \
  --data_dir /export/b01/zyli/data/joci_copa_2choice_new/3 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir joci_2choice_output










































CUDA_VISIBLE_DEVICES=$(free-gpu) python run_mnli_1.py \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --do_train \
  --do_eval \
  --do_test \
  --do_prediction 0 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new_all/1 \
  --train_batch_size 50 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 100 \
  --do_margin_loss 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output



408987
2722
2759
2740

# best_eval_acc1: 0.9033798677443057 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc2: 0.9155491119971004 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc3: 0.9759124087591241 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py

best_eval_acc1: 0.9019103600293902 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py
best_eval_acc2: 0.9238854657484595 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py
best_eval_acc3: 0.9751824817518249 32436 0.2 1 2e-05 bert-base-uncased run_mnli_1.py


                                                                                       
# best_eval_acc1: 0.8890521675238795 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc2: 0.920985864443639 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc3: 0.9733576642335766 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py

# best_eval_acc1: 0.9103600293901543 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc2: 0.9119246103660746 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py
# best_eval_acc3: 0.9777372262773723 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py

best_eval_acc1: 0.9088905216752388 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py
best_eval_acc2: 0.9227981152591519 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py
best_eval_acc3: 0.977007299270073 32436 0.2 0 2e-05 bert-base-uncased run_mnli_1.py






CUDA_VISIBLE_DEVICES=$(free-gpu) python run_mnli_2.py \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --do_train \
  --do_eval \
  --do_test \
  --do_prediction 0 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new_all/2 \
  --train_batch_size 50 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 100 \
  --do_margin_loss 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output

272649
2722
2759

# best_eval_acc1: 0.8802351212343865 32436 0.2 1 2e-05 bert-base-uncased run_mnli_2.py
# best_eval_acc2: 0.8999637549836897 32436 0.2 1 2e-05 bert-base-uncased run_mnli_2.py

# best_eval_acc1: 0.8930933137398971 32436 0.2 1 2e-05 bert-base-uncased run_mnli_2.py
# best_eval_acc2: 0.9108372598767669 32436 0.2 1 2e-05 bert-base-uncased run_mnli_2.py

best_eval_acc1: 0.8952975753122704 32436 0.2 1 2e-05 bert-base-uncased run_mnli_2.py
best_eval_acc2: 0.9148242116708952 32436 0.2 1 2e-05 bert-base-uncased run_mnli_2.py

# best_eval_acc1: 0.8997060984570169 32436 0.2 0 2e-05 bert-base-uncased run_mnli_2.py
# best_eval_acc2: 0.9072127582457412 32436 0.2 0 2e-05 bert-base-uncased run_mnli_2.py

# best_eval_acc1: 0.9048493754592212 32436 0.2 0 2e-05 bert-base-uncased run_mnli_2.py
# best_eval_acc2: 0.9195360637912287 32436 0.2 0 2e-05 bert-base-uncased run_mnli_2.py

best_eval_acc1: 0.9066862601028656 32436 0.2 0 2e-05 bert-base-uncased run_mnli_2.py
best_eval_acc2: 0.920985864443639 32436 0.2 0 2e-05 bert-base-uncased run_mnli_2.py
# -

CUDA_VISIBLE_DEVICES=$(free-gpu) python run_mnli_3.py \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --do_train \
  --do_eval \
  --do_test \
  --do_prediction 0 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new_all/3 \
  --train_batch_size 50 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 100 \
  --do_margin_loss 0 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output

142291
65745
64613

# best_eval_acc1: 0.8666818769488174 32436 0.2 1 2e-05 bert-base-uncased run_mnli_3.py
# best_eval_acc2: 0.8859362666955566 32436 0.2 1 2e-05 bert-base-uncased run_mnli_3.py

best_eval_acc1: 0.8752148452353791 32436 0.2 1 2e-05 bert-base-uncased run_mnli_3.py
best_eval_acc2: 0.8840481017751846 32436 0.2 1 2e-05 bert-base-uncased run_mnli_3.py

# best_eval_acc1: 0.876249144421629 32436 0.2 1 2e-05 bert-base-uncased run_mnli_3.py
# best_eval_acc2: 0.8819123086685342 32436 0.2 1 2e-05 bert-base-uncased run_mnli_3.py


# best_eval_acc1: 0.869161152939387 32436 0.2 0 2e-05 bert-base-uncased run_mnli_3.py
# best_eval_acc2: 0.8902388064321421 32436 0.2 0 2e-05 bert-base-uncased run_mnli_3.py

best_eval_acc1: 0.8763556163966841 32436 0.2 0 2e-05 bert-base-uncased run_mnli_3.py
best_eval_acc2: 0.8826397164657267 32436 0.2 0 2e-05 bert-base-uncased run_mnli_3.py

# best_eval_acc1: 0.8768575557076583 32436 0.2 0 2e-05 bert-base-uncased run_mnli_3.py
# best_eval_acc2: 0.8786776654852738 32436 0.2 0 2e-05 bert-base-uncased run_mnli_3.py


# 密度图绘制

1:
CUDA_VISIBLE_DEVICES=$(free-gpu) python dump_examples_mnli.py \
  --bert_model /export/b01/zyli/data/mnli_copa_2choice_new_all/1/model2_0 \
  --do_margin_loss 0 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new_all/1 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output


CUDA_VISIBLE_DEVICES=$(free-gpu) python dump_examples_mnli.py \
  --bert_model /export/b01/zyli/data/mnli_copa_2choice_new_all/1/model2_1 \
  --do_margin_loss 1 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new_all/1 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output



2:
CUDA_VISIBLE_DEVICES=$(free-gpu) python dump_examples_mnli.py \
  --bert_model /export/b01/zyli/data/mnli_copa_2choice_new_all/2/model2_0 \
  --do_margin_loss 0 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new_all/2 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output


CUDA_VISIBLE_DEVICES=$(free-gpu) python dump_examples_mnli.py \
  --bert_model /export/b01/zyli/data/mnli_copa_2choice_new_all/2/model2_1 \
  --do_margin_loss 1 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new_all/2 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output



3:
CUDA_VISIBLE_DEVICES=$(free-gpu) python dump_examples_mnli.py \
  --bert_model /export/b01/zyli/data/mnli_copa_2choice_new_all/3/model1_0 \
  --do_margin_loss 0 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new_all/3 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output


CUDA_VISIBLE_DEVICES=$(free-gpu) python dump_examples_mnli.py \
  --bert_model /export/b01/zyli/data/mnli_copa_2choice_new_all/3/model1_1 \
  --do_margin_loss 1 \
  --data_dir /export/b01/zyli/data/mnli_copa_2choice_new_all/3 \
  --do_train \
  --do_eval \
  --do_prediction 0 \
  --do_lower_case \
  --seed 32436 \
  --margin 0.2 \
  --train_batch_size 200 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 50 \
  --output_dir mnli_2choice_output




# COPA
# log
MARGIN: 0.37 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.98max epoch: 1    [6.48 6.98 6.76 5.44 4.32]
# margin
MARGIN: 0.37 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 7.08 max epoch: 2    [6.74 6.64 7.08 6.52 4.56]

CUDA_VISIBLE_DEVICES=$(free-gpu) python run_copa.py \
  --bert_model /home/zyli/github/copa/models/bert-base-uncased \
  --do_lower_case \
  --seed 6776 \
  --margin 0.37 \
  --l2_reg 0.02 \
  --do_train \
  --do_test \
  --do_margin_loss 1 \
  --data_dir /export/b01/zyli/data/copa_data \
  --train_batch_size 40 \
  --eval_batch_size 100 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 57 \
  --output_dir copa_output

CUDA_VISIBLE_DEVICES=$(free-gpu) python run_copa.py \
  --bert_model bert-large-uncased \
  --do_lower_case \
  --seed 6776 \
  --margin 0.31 \
  --l2_reg 0.001 \
  --do_train \
  --do_eval \
  --do_test \
  --do_margin_loss 1 \
  --data_dir /export/b01/zyli/data/copa_data \
  --train_batch_size 10 \
  --eval_batch_size 20 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --max_seq_length 51 \
  --output_dir copa_output


CUDA_VISIBLE_DEVICES=$(free-gpu) python run_copa.py \
  --bert_model /home/zyli/github/copa/models/bert-base-uncased \
  --do_lower_case \
  --seed 6776 \
  --margin 0.31 \
  --l2_reg 0.01 \
  --do_train \
  --do_eval \
  --do_test \
  --do_margin_loss 0 \
  --data_dir /export/b01/zyli/data/copa_data \
  --train_batch_size 10 \
  --eval_batch_size 100 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --max_seq_length 30 \
  --output_dir copa_output

# 0.78 15526
# 0.794 6776 0.31

# 在30开发集上调参
# 0.61 0.833 1 6776 0.31 0 train_batch_size=20
# 0.752 0.867 224 6776 0.31 1 train_batch_size=20

# 在50开发集上调参,明显更好 /home/zyli/github/copa/models/bert-base-uncased
0.744 0.82 66 6776 0.31 0 10 30
0.76 0.78 77 6776 0.31 1 10 51

# 在50开发集上调参MNLI: /home/zyli/github/copa/examples/mnli_output/model2
# 0.734 0.9 280 6776 0.31 0
# 0.736 0.88 777 6776 0.31 1

# 在50开发集上调参MNLI_copa: /home/zyli/github/copa/examples/mnli_copa_output/model2
# 0.704 0.82 222 6776 0.31 0
# 0.686 0.74 721 6776 0.31 1


#cross validation for COPA







# L2_REG: 0.01 max_accuracy: 6.78 max epoch: 2    [6.38 6.64 6.78 6.78 6.7 ]
*# L2_REG: 0.02 max_accuracy: 6.78 max epoch: 2    [6.38 6.64 6.78 6.78 6.72]
# L2_REG: 0.03 max_accuracy: 6.8 max epoch: 3    [6.38 6.66 6.76 6.8  6.7 ]
# 

# margin: 0.2 max_accuracy: 6.739999999999999 max epoch: 2    [6.56 6.68 6.74 6.64 6.44]
# margin: 0.21 max_accuracy: 6.720000000000001 max epoch: 2    [6.52 6.7  6.72 6.72 6.3 ]
# margin: 0.22 max_accuracy: 6.779999999999999 max epoch: 3    [6.5  6.76 6.7  6.78 6.42]
# margin: 0.23 max_accuracy: 6.8 max epoch: 1    [6.5  6.8  6.76 6.76 6.48]
# margin: 0.24 max_accuracy: 6.82 max epoch: 1    [6.5  6.82 6.72 6.74 6.52]
# margin: 0.25 max_accuracy: 6.78 max epoch: 1    [6.52 6.78 6.78 6.76 6.42]
# margin: 0.26 max_accuracy: 6.9 max epoch: 3    [6.48 6.86 6.86 6.9  6.5 ]
# margin: 0.27 max_accuracy: 6.9399999999999995 max epoch: 1    [6.52 6.94 6.86 6.92 6.42]
# margin: 0.28 max_accuracy: 6.8999999999999995 max epoch: 3    [6.52 6.86 6.84 6.9  6.5 ]
# margin: 0.29 max_accuracy: 6.9 max epoch: 3    [6.56 6.88 6.84 6.9  6.52]
# margin: 0.30 max_accuracy: 6.920000000000001 max epoch: 1    [6.54 6.92 6.84 6.82 6.54]
# margin: 0.31 max_accuracy: 6.9 max epoch: 1    [6.56 6.9  6.82 6.84 6.58]
# margin: 0.32 max_accuracy: 6.88 max epoch: 1    [6.56 6.88 6.74 6.82 6.52]
# margin: 0.33 max_accuracy: 6.8199999999999985 max epoch: 1    [6.58 6.82 6.66 6.8  6.6 ]
# margin: 0.34 max_accuracy: 6.860000000000001 max epoch: 1    [6.62 6.86 6.68 6.76 6.44]
# margin: 0.35 max_accuracy: 6.9 max epoch: 1    [6.6  6.9  6.76 6.74 6.66]
# margin: 0.36 max_accuracy: 6.92 max epoch: 1    [6.58 6.92 6.78 6.84 6.48]
*# margin: 0.37 max_accuracy: 6.94 max epoch: 3    [6.54 6.92 6.8  6.94 6.38]
# margin: 0.38 max_accuracy: 6.880000000000001 max epoch: 2    [6.58 6.86 6.88 6.82 6.62]
# margin: 0.39 max_accuracy: 6.9 max epoch: 1    [6.64 6.9  6.84 6.84 6.52]
# -


# TRAIN_BATCH_SIZE: 10 max_accuracy: 6.9399999999999995 max epoch: 2    [6.84 6.9  6.94 6.8  5.52]
# TRAIN_BATCH_SIZE: 10 max_accuracy: 6.779999999999999 max epoch: 2    [6.7  6.68 6.78 6.72 5.04]
# TRAIN_BATCH_SIZE: 20 max_accuracy: 6.7799999999999985 max epoch: 3    [6.48 6.68 6.76 6.78 5.72]
# TRAIN_BATCH_SIZE: 30 max_accuracy: 6.720000000000001 max epoch: 2    [6.44 6.66 6.72 6.54 6.22]
*# TRAIN_BATCH_SIZE: 40 max_accuracy: 6.959999999999999 max epoch: 3    [6.62 6.88 6.9  6.96 4.52]
# TRAIN_BATCH_SIZE: 50 max_accuracy: 6.739999999999999 max epoch: 3    [6.44 6.72 6.7  6.74 6.56]
# TRAIN_BATCH_SIZE: 60 max_accuracy: 6.779999999999999 max epoch: 1    [6.38 6.78 6.72 6.72 6.2 ]

# MAX_SEQ_LENGTH: 45 max_accuracy: 6.78 max epoch: 1    [6.74 6.78 6.7  6.76 6.48]
# MAX_SEQ_LENGTH: 47 max_accuracy: 6.639 max epoch: 2    [6.58 6.62 6.64 6.62 6.36]
# MAX_SEQ_LENGTH: 49 max_accuracy: 6.699 max epoch: 1    [6.56 6.7  6.64 6.58 5.86]
# MAX_SEQ_LENGTH: 51 max_accuracy: 6.7 max epoch: 3    [6.6  6.66 6.68 6.7  5.1 ]
# MAX_SEQ_LENGTH: 53 max_accuracy: 6.72 max epoch: 3    [6.6  6.58 6.6  6.72 5.26]
# MAX_SEQ_LENGTH: 55 max_accuracy: 6.699 max epoch: 2    [6.52 6.66 6.7  6.64 6.24]
*# MAX_SEQ_LENGTH: 57 max_accuracy: 6.939 max epoch: 2    [6.68 6.92 6.94 6.92 5.88]
# MAX_SEQ_LENGTH: 59 max_accuracy: 6.74 max epoch: 3    [6.64 6.7  6.7  6.74 6.02]
# MAX_SEQ_LENGTH: 61 max_accuracy: 6.60 max epoch: 1    [6.54 6.6  6.6  6.52 5.36]


# LR: 1e-05 max_accuracy: 6.859 max epoch: 1    [6.8  6.86 6.84 6.8  5.72]
# LR: 2e-05 max_accuracy: 7.06 max epoch: 2    [6.9  7.02 7.06 6.82 4.08]
LR: 3e-05 max_accuracy: 7.08 max epoch: 2    [6.84 6.94 7.08 5.68 4.6 ]


SEED: 6776 max_accuracy: 6.819 max epoch: 2    [6.6  6.78 6.82 6.68 6.  ]
# SEED: 10071 max_accuracy: 6.86 max epoch: 2    [6.38 6.74 6.86 6.46 4.2 ]
# SEED: 21451 max_accuracy: 6.76 max epoch: 1    [6.42 6.76 6.68 6.76 5.92]
# SEED: 5640 max_accuracy: 6.02 max epoch: 1    [5.7  6.02 5.84 5.56 4.52]
# SEED: 9918 max_accuracy: 6.399 max epoch: 2    [6.32 6.38 6.4  5.98 4.84]
# SEED: 3054 max_accuracy: 6.72 max epoch: 1    [6.46 6.72 6.72 6.72 5.22]
# SEED: 26288 max_accuracy: 6.28 max epoch: 1    [6.   6.28 6.26 6.   4.58]
# SEED: 17370 max_accuracy: 6.68 max epoch: 1    [6.22 6.68 6.62 6.52 4.66]
# SEED: 14683 max_accuracy: 6.64 max epoch: 3    [6.56 6.54 6.54 6.64 5.68]
# SEED: 24196 max_accuracy: 5.64 max epoch: 0    [5.64 5.64 5.54 5.2  5.44]
# SEED: 27794 max_accuracy: 6.42 max epoch: 1    [6.36 6.42 6.42 6.22 4.9 ]
# SEED: 29135 max_accuracy: 6.16 max epoch: 2    [5.96 5.94 6.16 5.84 5.66]
# SEED: 29602 max_accuracy: 6.66 max epoch: 0    [6.66 6.62 6.54 6.32 4.88]
# SEED: 5074 max_accuracy: 6.42 max epoch: 2    [6.22 6.38 6.42 6.12 5.04]
# SEED: 673 max_accuracy: 6.58 max epoch: 2    [6.24 6.56 6.58 6.38 5.6 ]
# SEED: 5516 max_accuracy: 6.68 max epoch: 2    [6.38 6.54 6.68 6.54 5.44]
# SEED: 6844 max_accuracy: 5.959 max epoch: 1    [5.78 5.96 5.84 5.92 5.78]
# SEED: 18341 max_accuracy: 6.56 max epoch: 1    [6.   6.56 6.56 6.38 5.74]
# SEED: 3808 max_accuracy: 6.1 max epoch: 1    [5.96 6.1  6.02 5.6  5.24]
# SEED: 13078 max_accuracy: 6.34 max epoch: 2    [5.96 6.22 6.34 5.88 5.04]
# SEED: 18352 max_accuracy: 6.54 max epoch: 0    [6.54 6.28 6.48 6.34 5.48]



# 将最好的2个参数进行组合，得到最好的交叉验证参数
# 
# log-loss

# MARGIN: 0.27 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.68 max epoch: 1    [6.46 6.68 6.62 6.34 4.32]
# MARGIN: 0.27 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.76max epoch: 1    [5.92 6.76 6.58 5.32 4.04]
# MARGIN: 0.27 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.659 max epoch: 2    [6.56 6.62 6.66 5.94 4.62]
# MARGIN: 0.27 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.72max epoch: 2    [6.44 6.64 6.72 5.06 4.6 ]
# MARGIN: 0.27 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.54 max epoch: 1    [5.76 6.54 6.48 5.42 3.9 ]
# MARGIN: 0.27 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.34max epoch: 2    [5.36 6.14 6.34 4.58 4.8 ]
# MARGIN: 0.27 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.76 max epoch: 2    [6.66 6.6  6.76 5.34 4.36]
# MARGIN: 0.27 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.92max epoch: 1    [6.52 6.92 6.72 5.42 4.26]
# MARGIN: 0.27 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.8 max epoch: 2    [6.36 6.62 6.8  6.54 4.24]
# MARGIN: 0.27 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.459 max epoch: 1    [5.8  6.46 6.42 5.34 4.24]
# MARGIN: 0.27 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.6 max epoch: 2    [6.58 6.58 6.6  5.86 4.78]
# MARGIN: 0.27 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.72max epoch: 2    [6.44 6.62 6.72 5.   4.46]
# MARGIN: 0.27 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.56 max epoch: 2    [5.92 6.44 6.56 5.08 4.04]
# MARGIN: 0.27 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 5.98max epoch: 1    [5.24 5.98 5.82 5.22 4.38]
# MARGIN: 0.27 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.70 max epoch: 2    [6.62 6.42 6.7  5.2  4.6 ]
# MARGIN: 0.27 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.98max epoch: 1    [6.48 6.98 6.76 5.44 4.32]
# MARGIN: 0.37 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.68 max epoch: 1    [6.46 6.68 6.62 6.34 4.32]
# MARGIN: 0.37 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.76max epoch: 1    [5.92 6.76 6.58 5.32 4.04]
# MARGIN: 0.37 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.659 max epoch: 2    [6.56 6.62 6.66 5.94 4.62]
# MARGIN: 0.37 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.72max epoch: 2    [6.44 6.64 6.72 5.06 4.6 ]
# MARGIN: 0.37 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.54 max epoch: 1    [5.76 6.54 6.48 5.42 3.9 ]
# MARGIN: 0.37 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.34max epoch: 2    [5.36 6.14 6.34 4.58 4.8 ]
# MARGIN: 0.37 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.760 max epoch: 2    [6.66 6.6  6.76 5.34 4.36]
# MARGIN: 0.37 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.92max epoch: 1    [6.52 6.92 6.72 5.42 4.26]
# MARGIN: 0.37 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.8 max epoch: 2    [6.36 6.62 6.8  6.54 4.24]
# MARGIN: 0.37 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.459 max epoch: 1    [5.8  6.46 6.42 5.34 4.24]
# MARGIN: 0.37 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.6 max epoch: 2    [6.58 6.58 6.6  5.86 4.78]
# MARGIN: 0.37 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.72max epoch: 2    [6.44 6.62 6.72 5.   4.46]
# MARGIN: 0.37 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.56 max epoch: 2    [5.92 6.44 6.56 5.08 4.04]
# MARGIN: 0.37 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 5.98max epoch: 1    [5.24 5.98 5.82 5.22 4.38]
# MARGIN: 0.37 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.70 max epoch: 2    [6.62 6.42 6.7  5.2  4.6 ]
MARGIN: 0.37 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.98max epoch: 1    [6.48 6.98 6.76 5.44 4.32]

# margin-loss

# MARGIN: 0.27 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.84 max epoch: 2    [6.52 6.66 6.84 6.82 4.56]
# MARGIN: 0.27 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.76max epoch: 2    [6.26 6.74 6.76 6.68 3.84]
# MARGIN: 0.27 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.8 max epoch: 2    [6.48 6.72 6.8  6.56 4.42]
# MARGIN: 0.27 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.739 max epoch: 2    [6.12 6.62 6.74 6.34 4.28]
# MARGIN: 0.27 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.76 max epoch: 2    [5.96 6.42 6.76 6.14 4.2 ]
# MARGIN: 0.27 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.49 max epoch: 1    [5.98 6.5  6.48 5.7  4.62]
# MARGIN: 0.27 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.62 max epoch: 2    [6.36 6.52 6.62 6.06 4.86]
# MARGIN: 0.27 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.49 max epoch: 2    [6.22 6.42 6.5  5.84 4.32]
# MARGIN: 0.27 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.759 max epoch: 3    [6.5  6.64 6.64 6.76 4.62]
# MARGIN: 0.27 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.82max epoch: 3    [6.26 6.78 6.78 6.82 4.18]
# MARGIN: 0.27 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.819 max epoch: 2    [6.48 6.76 6.82 6.42 4.64]
# MARGIN: 0.27 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.64 max epoch: 2    [6.12 6.54 6.64 6.28 4.26]
# MARGIN: 0.27 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.659 max epoch: 2    [6.58 6.64 6.66 6.46 4.46]
# MARGIN: 0.27 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.479 max epoch: 2    [6.1  6.28 6.48 6.42 4.52]
# MARGIN: 0.27 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.50 max epoch: 2    [6.34 6.48 6.5  6.08 4.78]
# MARGIN: 0.27 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.54 max epoch: 2    [6.1  6.44 6.54 5.36 4.56]
MARGIN: 0.37 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 7.08 max epoch: 2    [6.74 6.64 7.08 6.52 4.56]
# MARGIN: 0.37 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.62 max epoch: 2    [6.06 6.52 6.62 6.5  4.36]
# MARGIN: 0.37 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.739 max epoch: 1    [6.56 6.74 6.72 6.48 4.6 ]
# MARGIN: 0.37 L2_REG: 0.02 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.66max epoch: 2    [6.58 6.52 6.66 6.02 4.36]
# MARGIN: 0.37 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.36 max epoch: 1    [6.22 6.36 6.3  6.04 4.78]
# MARGIN: 0.37 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.0 max epoch: 1    [5.68 6.   5.88 5.7  4.5 ]
# MARGIN: 0.37 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.559 max epoch: 2    [6.48 6.52 6.56 6.04 4.62]
# MARGIN: 0.37 L2_REG: 0.02 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.979 max epoch: 2    [6.4  6.8  6.98 5.28 4.68]
# MARGIN: 0.37 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.99 max epoch: 2    [6.62 6.72 7.   6.54 4.16]
# MARGIN: 0.37 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.74max epoch: 1    [6.12 6.74 6.74 6.68 3.96]
# MARGIN: 0.37 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.74 max epoch: 2    [6.56 6.74 6.74 6.6  4.3 ]
# MARGIN: 0.37 L2_REG: 0.03 LR: 2e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.66max epoch: 2    [6.58 6.58 6.66 6.34 4.2 ]
# MARGIN: 0.37 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 6776 max_accuracy: 6.46 max epoch: 1    [6.06 6.46 6.22 6.08 4.08]
# MARGIN: 0.37 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 10 SEED: 10071 max_accuracy: 6.28max epoch: 2    [5.94 6.2  6.28 5.56 4.52]
# MARGIN: 0.37 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 6776 max_accuracy: 6.539 max epoch: 2    [6.48 6.48 6.54 5.6  4.18]
# MARGIN: 0.37 L2_REG: 0.03 LR: 3e-05 TRAIN_BATCH_SIZE: 40 SEED: 10071 max_accuracy: 6.899 max epoch: 2    [6.28 6.82 6.9  5.34 4.16]


# 使用最好参数在整个dev上训练，在test上测试：
~/github/copa/examples(master*) » CUDA_VISIBLE_DEVICES=$(free-gpu) python run_copa.py \
  --bert_model /home/zyli/github/copa/models/bert-base-uncased \
  --do_lower_case \
  --seed 10071 \
  --margin 0.37 \
  --l2_reg 0.02 \
  --do_train \
  --do_test \
  --do_margin_loss 0 \
  --data_dir /export/b01/zyli/data/copa_data \
  --train_batch_size 40 \
  --eval_batch_size 100 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 57 \
  --output_dir copa_output

Epoch: 0
Epoch: 1
Epoch: 2
0.734 0.0
0.734 0.0 0 10071 0.37 0 40 57 /home/zyli/github/copa/models/bert-base-uncased run_copa.py


CUDA_VISIBLE_DEVICES=$(free-gpu) python run_copa.py \
  --bert_model /home/zyli/github/copa/models/bert-base-uncased \
  --do_lower_case \
  --seed 6776 \
  --margin 0.37 \
  --l2_reg 0.02 \
  --do_train \
  --do_test \
  --do_margin_loss 1 \
  --data_dir /export/b01/zyli/data/copa_data \
  --train_batch_size 40 \
  --eval_batch_size 100 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 57 \
  --output_dir copa_output

Epoch: 0
Epoch: 1
Epoch: 2
0.754 0.0
0.754 0.0 0 6776 0.37 1 40 57 /home/zyli/github/copa/models/bert-base-uncased run_copa.py



