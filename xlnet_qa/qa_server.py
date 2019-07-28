#!/usr/bin/env python
import plyvel
from tqdm import tqdm
from nltk import sent_tokenize
from nlptools.text import TFIDF
from nlptools.utils import zloads, zdumps

class QAServer:
    """
        QA restful server
    """
    def __init__(self, filepath, tokenizer, url_rule="/api/query", port=5002):
        """
            Input:
                - filepath: txt file path, support .gzip, .bzip2, and .txt file
                - tokenizer: tokenizer from pytorch_transformers
        """
        self.tokenizer = tokenizer
        self.tfidf = None
        self.build_db(filepath)
        self.build_index(filepath)

    def build_db(self, filepath):
        """
            save txt to LevelDB

            Input:
                - filepath: txt file path, support .gzip, .bzip2, and .txt file
        """
        cached_contents = filepath + ".contents"
        if os.path.exists(cached_contents):
            return
        self.contents = plyvel.DB(cached_contents, create_if_missing=True)
        ext = os.path.splitext(filepath)

        if ext == ".gz":
            import gzip
            open_func = gzip.open
        elif ext == ".bz2":
            open_func = bz2.open
        else:
            open_func = open

        with open_func(filepath) as f, self.contents.write_batch() as wb:
            totN = 0
            for i, l in tqdm(enumerate(f)):
                for j, sent in enumerate(sent_tokenize(l)):
                    token_ids = self.tokenizer.encode(j)
                    wb.put(bytes(totN), zdumps({"sentence": sent, "ids": token_ids}))
                    totN += 1
            wb.put(b"total", zdumps({"N": totN, "Ndocs": i+1}))

    def build_index(self, filepath):
        """
            build index

            Input:
                - filepath: txt file path, support .gzip, .bzip2, and .txt file
        """
        cached_index = filepath + '.index'
        self.tfidf = TFIDF(vocab_size=self.tokenizer.vocab_size, cached_index=cached_index)
        totN = zloads(self.contents.get(b'total'))["N"]
        def data_iter():
            with self.contents.snapshot() as sn:
                for i in tqdm(range(totN)):
                    yield zloads(sn.get(bytes(i)))["ids"]
        self.tfidf.load_index(corpus_ids=data_iter, corpus_len=totN)


