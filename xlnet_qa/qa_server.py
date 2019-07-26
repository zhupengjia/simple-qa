#!/usr/bin/env python
from nltk import sent_tokenize
from nlptools.text import TFIDF
from nlptools.utils import zload, zdump
import plyvel

class QAServer:
    """
        QA restful server
    """
    def __init__(self, ziptxt, tokenizer, cached_index, url_rule="/api/query", port=5002):
        """
            Input:
                - ziptxt: compressed txt file via gzip
                - tokenizer: tokenizer from pytorch_transformers
        """
        self.tokenizer = tokenizer
        self.tfidf = TFIDF(vocab_size=self.tokenizer.vocab_size, cached_index=cached_index)
        self.cached_contents = cached_index + ".contents"

    def build_index(self, ziptxt):
        """
            Build index

            Input:
                - ziptxt: compressed txt file via gzip
        """
        if os.path.exists(self.cached_contents):
            return
        self.contents = plyvel.DB(self.cached_contents, create_if_missing=True)
        num_lines = sum(1 for line in gzip.open(ziptxt))
        with gzip.open(ziptxt) as f:
            for i, l in enumerate(f):
                for j, sent in enumerate(sent_tokenize(l)):
                    token_ids = self.tokenizer.encode(j)




