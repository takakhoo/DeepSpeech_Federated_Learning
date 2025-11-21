# example: run on gpu 0, batch start index 0, end index inf, min duration 0, max duration 1000
# normal experiment on lr 0.5
# git checkout a7bd5a0   
CUDA_VISIBLE_DEVICES="0" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 0 --batch_max_dur 1000
CUDA_VISIBLE_DEVICES="1" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000
CUDA_VISIBLE_DEVICES="2" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 2000 --batch_max_dur 3000
CUDA_VISIBLE_DEVICES="3" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 3000 --batch_max_dur 4000

# lr experiment 0.5 1 0.1 0.01 0.001
# git checkout a7bd5a0   
CUDA_VISIBLE_DEVICES="4" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 1.0
CUDA_VISIBLE_DEVICES="4" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 0.5
CUDA_VISIBLE_DEVICES="5" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 0.1
CUDA_VISIBLE_DEVICES="6" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 0.01
CUDA_VISIBLE_DEVICES="7" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 0.001
CUDA_VISIBLE_DEVICES="0" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 5.0
CUDA_VISIBLE_DEVICES="1" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 10.0
# top grad percentage experiment 100% 90% 80% 70% 60% 50%
# git checkout a7bd5a0   
CUDA_VISIBLE_DEVICES="5" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 0.5 --top_grad_percentage 0.2
CUDA_VISIBLE_DEVICES="6" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 0.5 --top_grad_percentage 0.4
CUDA_VISIBLE_DEVICES="7" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 0.5 --top_grad_percentage 0.6
CUDA_VISIBLE_DEVICES="0" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 0.5 --top_grad_percentage 0.8


# cosine + l2 experiment
CUDA_VISIBLE_DEVICES="0" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 0.5 --distance_function cosine+l2 --distance_function_weight 0.9
CUDA_VISIBLE_DEVICES="1" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 0.5 --distance_function cosine+l2 --distance_function_weight 0.8
CUDA_VISIBLE_DEVICES="2" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 0.5 --distance_function cosine+l2 --distance_function_weight 0.7
CUDA_VISIBLE_DEVICES="3" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 0.5 --distance_function cosine+l2 --distance_function_weight 0.6
CUDA_VISIBLE_DEVICES="4" python src/reconstruct_ds1_run_many_sample.py --batch-start 0 --batch-end 100 --batch_min_dur 1000 --batch_max_dur 2000 --lr 0.5 --distance_function cosine+l2 --distance_function_weight 0.5