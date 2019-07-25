#!/usr/bin/env python
from pytorch_transformers import BERTTokenizer

class QAServer:
    """
        QA restful server
    """
    def __init__(self, tokenizer_name="bert-base-uncased", url_rule="/api/query", port=5002):
        self.tokenizer = BERTTokenizer.from_pretrained(tokenizer_name)
        self.vocab =self.tokenizer.vocab

    def build_index(self, ziptxt):
        """
            Build index

            Input:
                - ziptxt: compressed txt file via gzip
        """
        with gzip.open(ziptxt) as f:
            pass



