# Simple QA

Question answering using pretrained models based on Xapian and XLNet

* Usage:

    - For anyone want to try, I suggest to use docker.
        - For CPU version, run to pull image:
            
            ```shell
                docker pull zhupengjia/simple-qa:cpu
            ```

        - For GPU version, run to pull image:
            
            ```shell
                docker pull zhupengjia/simple-qa:gpu
            ```
        
        - Download pretrained model from following link to your directory:, then decompress:

            https://1drv.ms/u/s!AnzH-f0hZoPctxAyoLAyA-b0ab6A?e=F2ks1i

            Then decompress:

            ```shell
                tar xzvf squad2_xlnet.tar
            ```

        - Make sure you have data that in the format of .pdf, .txt, .gzip, .bzip2

        - Then try to run manually:
            
            Assume your directory contains model and data

            ```shell
                docker run -d -v YOURDIRECTORY:/opt/chatbot/data --name simple_qa zhupengjia/simple-qa:cpu tail -f /dev/null
            ```
                
            For GPU version, please make sure you have installed nvidia-container-runtime, and has the following item in your /etc/docker/daemon.json file:
            ```shell
                "runtimes": {
                    "nvidia": {
                        "path": "/usr/bin/nvidia-container-runtime",
                            "runtimeArgs": []
                    }
                }
            ```
            Then create container as:

            ```shell
                docker run -d --runtime nvidia -v YOURDIRECTORY:/opt/chatbot/data --name simple_qa zhupengjia/simple-qa:cpu tail -f /dev/null
            ```

            Then manually do:
            
            ```shell
                docker exec -it simple_qa bash
                python3 interact.py  -m data/checkpoint-7900 --returnrelate --backend shell data/sample.txt
            ```
            Have fun

        - If you want to run a restfulapi:

            ```shell
                cd docker
            ```

            modify docker-compose.yml for docker-compose-gpu.yml for image, environment, volumes

            ```shell
                docker-compose up -d
            ```
            or

            ```shell
                docker-compose -f docker-compose-gpu.yml up -d
            ```

    - If you want to train model by yourself:

        ```shell
        cd ANYDIRECTORY
        git clone https://github.com/huggingface/transformers.git
        cd transformers/examples
        python run_squad.py --do_lower_case --version_2_with_negative --model_type xlnet --model_name_or_path xlnet-large-cased --do_train --do_eval --train_file data/train-v2.0.json --predict_file data/dev-v2.0.json --learning_rate 3e-6 --num_train_epochs 12 --max_seq_length 384 --doc_stride 128 --output_dir ./finetuned_squad_xlnet --per_gpu_eval_batch_size 2 --per_gpu_train_batch_size 2 --save_steps 100 --fp16 --gradient_accumulation_steps 100 --overwrite_output_dir --do_lower_case
        ```

    - If you want to run in local machine::
        
        Please make sure you have installed python-xapian. If you want to parse pdf file, please make sure you have installed poppler-utils

        - run:

            ```shell
            python interact.py -m MODELPATH --returnrelated --scorelimit 0.2 textfile_path
            ```

