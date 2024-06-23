export HYDRA_FULL_ERROR=1


seed_list=( \
# "1" \
"42" \
# "3407" \
)
batch_size=1
device='0'
task_name=scicite
rerank_method=random  # human, random
model_name="model/Llama-2-70b-chat-hf"
sample_num=5

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

# 使用for循环遍历随机种子
for seed in ${seed_list[@]}  # 1, 42, 3407
do
  echo "Running experiment with seed $seed"

  output_file=output/pred_data/${task_name}/${model}/manual+${rerank_method}_seed${seed}_bs${batch_size}.json

  CUDA_VISIBLE_DEVICES=${device} python inference/one_stage.py output_file=${output_file} \
                                rand_seed=${seed} \
                                batch_size=${batch_size} \
                                task_name=${task_name} \
                                rerank_method=${rerank_method} \
                                sample_num=${sample_num} \
                                model_name=${model_name}
done