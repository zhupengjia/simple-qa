#!/usr/bin/env python
import os, nltk, string, torch, shutil
from tqdm import tqdm
from pytorch_transformers import XLNetForQuestionAnswering
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import TEXT, Schema
from whoosh import qparser, index
from .squad2_reader import SQuAD2Reader

class QAServer:
    """
        QA restful server
    """
    def __init__(self,
                 file_path,
                 model_path,
                 model_name = "xlnet-large-cased",
                 max_seq_len = 384,
                 doc_stride = 128,
                 max_query_len = 64,
                 device = "cuda:0",
                 recreate=False,
                 **args):
        """
            Input:
                - file_path: txt file path, support .gzip, .bzip2, and .txt file
                - model_path: path of model, must be a directory and contains file "pytorch_model.bin" and "config.json"
                - tokenizer: tokenizer from pytorch_transformers
                - recreate: bool, True will force recreate db, default is False
        """
        self.reader = SQuAD2Reader(tokenizer_name = model_name,
                                  max_seq_len = max_seq_len,
                                  doc_stride = doc_stride,
                                  max_query_len = max_query_len,
                                  is_training = False)
        if not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        self.translator = str.maketrans('', '', string.punctuation) # remove punctuation

        self._build_index(file_path, recreate)
        self._load_model(model_path, self.device)

    def _get_cursor(self, dbfile):
        db = sqlite3.connect(dbfile)
        cursor = db.cursor()
        return db, cursor

    def _build_index(self, filepath, recreate=False):
        """
            save txt to LevelDB

            Input:
                - filepath: txt file path, support .gzip, .bzip2, and .txt file
                - recreate: bool, True will force recreate db, default is False
        """
        cached_index = filepath + ".index"

        if os.path.exists(cached_index):
            if recreate:
                shutil.rmtree(cached_index)
        else:
            recreate = True

        stem_ana = StemmingAnalyzer()
        self.schema = Schema(content=TEXT(stored=True, analyzer=stem_ana))

        if not recreate:
            self.ix = index.open_dir(cached_index, indexname="qa_content")
        else:
            os.makedirs(cached_index)
            self.ix = index.create_in(cached_index, self.schema, indexname="qa_content")

            ext = os.path.splitext(filepath)[-1]
            if ext == ".bz2":
                import bz2
                open_func = bz2.open
            elif ext == ".gz":
                import gzip
                open_func = gzip.open
            else:
                open_func = open

            with open_func(filepath, mode="rt", encoding="utf-8") as f:
                totN, totP, totS= 0, 0, 0
                writer = self.ix.writer()
                for l in tqdm(f, desc="Building index", unit=" lines"):
                    l = l.strip()
                    if len(l) < 1 :
                        if totS > 0 : totP += 1
                        totS = 0
                        continue
                    for sent in nltk.sent_tokenize(l):
                        sent = sent.strip()
                        clean_sent = sent.translate(self.translator).lower().strip()
                        if len(clean_sent) < 1:
                            continue
                        writer.add_document(content=clean_sent)
                        totN += 1
                        totS += 1
                writer.commit()
        
        og = qparser.OrGroup.factory(0.9)
        self.parser = qparser.QueryParser("content", schema=ix.schema, group=og)
        self.parser.add_plugin(qparser.FuzzyTermPlugin())

    def _load_model(self, model_path, device):
        self.model = XLNetForQuestionAnswering.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def search(self, text, topN=5):
        """
            search related text

            Input:
                - text: string
                - topN: int, return topN results, default is 5
        """
        text = text.translate(self.translator).lower().strip()
        if len(text) < 1:
            return None
        query = parser.parse(text)
        
        results = searcher.search(query, limit=topN, terms=True)
        return [r['content'] for r in results]

    def __call__(self, question):
        related_texts = self.search(question)
        example, feature, dataset = self.reader(question, "\n".join(related_texts))
        dataset = tuple(t.to(self.device) for t in dataset)
        outputs = self.model(input_ids = dataset[0],
                        attention_mask = dataset[1],
                        token_type_ids = dataset[2],
                        cls_index = dataset[4],
                        p_mask = dataset[5]
                       )
        answer, score = self.reader.convert_output_to_answer(example, feature, outputs, self.model.config.start_n_top, self.model.config.end_n_top)
        return answer, score

