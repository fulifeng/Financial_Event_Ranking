# Financial Event Ranking
This is the implementation of our SIGIR 2021 paper: [Hybrid Learning to Rank for Financial Event Ranking](https://dl.acm.org/doi/10.1145/3404835.3462969). 

## Requirements
numpy==1.19.2
torch==1.4.0
tqdm==4.49.0
transformers==3.2.0

To create an environment with Conda:
```Bash
git clone https://github.com/fulifeng/Financial_Event_Ranking.git
cd Financial_Event_Ranking
conda create -n financial_event_ranking
conda activate financial_event_ranking
pip install -r requirements.txt
```

## Dataset
Please send an email to ___ if you need a copy of our data. 

## Training and Testing
#### Ret
* Training
```
cd Ret
python train_ret.py --lr 1e-05 --maxsteps 5000 --warmup_steps 100 --bsize 128 --accum 2 --do_eval_steps 100 --print_log_steps 100 --model_save_dir ./model_metal_ret
```
* Testing(just set --maxstep=0)
```
python train_ret.py --maxsteps=0 --model_to_test ./model_metal_ret_LR1e-05_BSIZE128 --result_save_file metal_ret_test.json
```

#### Cla_M
* Training
```
cd Cla_M
python train_clam.py --lr 1e-05 --maxsteps 5000 --warmup_steps 100 --bsize 128 --accum 2 --do_eval_steps 100 --print_log_steps 100 --model_save_dir ./model_metal_clam
```
* Testing(with --maxsteps=0)
```
python train_clam.py --maxsteps=0 --model_to_test ./model_metal_clam_LR1e-05_BSIZE128 --result_save_file metal_clam_test.json
```


#### HNB_CNN
* First obtain the result from Ret and Cla_M(with --augment) 
```
cd Ret
python train_ret.py --maxsteps=0 --model_to_test ./model_metal_ret_LR1e-05_BSIZE128 --result_save_file metal_ret_hybrid_test.json --augment
cd ../Cla_M
python train_clam.py --maxsteps=0 --model_to_test ./model_metal_clam_LR1e-05_BSIZE128 --result_save_file metal_clam_hybrid_test.json --augment
cd ..
```

* Training

```
cd hybrid_method
python train_h.py --lr 1e-04 --maxsteps 50000 --bsize 32 --accum 2 --do_eval_steps 100 --print_log_steps 100 --ret_result_file ../Ret/model_metal_ret_LR1e-05_BSIZE128/metal_ret_hybrid_test.json --clam_result_file ../Cla_M/model_metal_clam_LR1e-05_BSIZE128/metal_clam_hybrid_test.json --rerank_num 100 -- model_save_dir ./model_metal_hybrid
```

* Testing
```
python train_h.py --maxsteps 0 --model_to_test ./model_metal_hybrid_LR0.0001_BSIZE32 --result_save_file metal_hybrid_test.json --ret_result_file ../Ret/model_metal_ret_LR1e-05_BSIZE128/metal_ret_hybrid_test.json --clam_result_file ../Cla_M/model_metal_clam_LR1e-05_BSIZE128/metal_clam_hybrid_test.json --rerank_num 100 
```






