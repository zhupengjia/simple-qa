# Simple QA

Question answering using pretrained models based on Xapian and Distilbert

* Usage:

    - For anyone want to try, I suggest to use docker.
        - Pull image:
            
            ```shell
                docker pull zhupengjia/simple-qa:distilbert
            ```

        - Make sure you have data that in the format of .pdf, .txt, .gzip, .bzip2

        - Then try to run manually:
            
            Please make sure your directory contains your data

            ```shell
                docker run -d -v YOURDIRECTORY:/opt/chatbot/data --name simple_qa zhupengjia/simple-qa:distilbert tail -f /dev/null
                docker exec -it simple_qa bash
                python3 interact.py --returnrelate --backend shell data/sample.txt
                have fun
            ```

        - If you want to run a restfulapi:
            
            ```shell
                cd docker
                modify docker-compose.yml for image, environment, volumes
                docker-compose up -d
            ```

    - If you want to train model by yourself, please check the repository: 
        - https://github.com/huggingface/transformers.git

    - If you want to run in local machine::
        
        Please make sure you have installed python-xapian. If you want to parse pdf file, please make sure you have installed poppler-utils

        - run:

            ```shell
            python interact.py --returnrelated --scorelimit 0.2 textfile_path
            ```

