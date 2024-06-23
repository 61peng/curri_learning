export HYDRA_FULL_ERROR=1


seed_list=( \
"1" \
"42" \
"3407" \
)
batch_size=4
task_name=scinli
prerank_method=topk  # topk, votek
coverage=false
rerank_method=mdl  # ppl_label, ppl_sentence, length, similarity, random, mdl, entropy
model_name="model/Mixtral-8x7B-Instruct-v0.1"
sample_num=5
max_new_token=512
# 模型简称
if [ ${model_name} = 'model/Mixtral-8x7B-Instruct-v0.1' ];
then
  model='mixtral'
elif [ ${model_name} = 'model/Llama-2-70b-chat-hf' ];
then
  model='llama2-70b'
elif [ ${model_name} = 'model/Qwen1.5-72B-Chat' ];
then
  model='qwen1.5-72b'
fi

# 输入文件
if [ ${rerank_method} = 'mdl' ] || [ ${rerank_method} = 'entropy' ];
then
  input_file=output/retrived_data/${task_name}/rerank_${rerank_method}_${model}.json
else
  input_file=output/retrived_data/${task_name}/retrieved_${prerank_method}.json
fi

# 使用for循环遍历随机种子
for seed in ${seed_list[@]}  # 1, 42, 3407
do
  echo "Running experiment with seed $seed"

  if [ ${coverage} = 'true' ];
  then
    output_file=output/pred_data/${task_name}/${model}/${prerank_method}c+${rerank_method}_seed${seed}_sample${sample_num}.json
  else
    output_file=output/pred_data/${task_name}/${model}/${prerank_method}+${rerank_method}_seed${seed}_bs${batch_size}.json
  fi

  python inference/two_stage.py output_file=${output_file} \
                                rand_seed=${seed} \
                                batch_size=${batch_size} \
                                task_name=${task_name} \
                                prerank_method=${prerank_method} \
                                rerank_method=${rerank_method} \
                                coverage=${coverage} \
                                sample_num=${sample_num} \
                                dataset_reader.dataset_path=${input_file} \
                                max_new_token=${max_new_token} \
                                model_name=${model_name}
done
