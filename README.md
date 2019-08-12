# Simple QA

Question answering using pretrained models based on Xapian and XLNet

* Usage:

    - Finetune model using pytorch-transformers:
    
    	```shell
		cd examples
		python run_squad.py --do_lower_case --version_2_with_negative --model_type xlnet --model_name_or_path /home/pzhu/data/qa/squad2_model --do_train --do_eval --train_file data/train-v2.0.json --predict_file data/dev-v2.0.json --learning_rate 3e-6 --num_train_epochs 12 --max_seq_length 384 --doc_stride 128 --output_dir ./finetuned_squad_xlnet --per_gpu_eval_batch_size 2 --per_gpu_train_batch_size 2 --save_steps 100 --fp16 --gradient_accumulation_steps 100 --overwrite_output_dir
    	```

    - run: 
    	```shell
    		python interact.py -m MODELPATH textfile_path
	```

