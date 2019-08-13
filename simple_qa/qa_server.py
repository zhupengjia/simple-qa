#!/usr/bin/env python
import os, nltk, torch, shutil, xapian
from tqdm import tqdm
from pytorch_transformers import XLNetForQuestionAnswering
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
                 recreate = False,
                 score_limit = 0.2,
                 return_relate = False,
                 **args):
        """
            Input:
                - file_path: txt file path, support .gzip, .bzip2, and .txt file
                - model_path: path of model, must be a directory and contains file "pytorch_model.bin" and "config.json"
                - tokenizer: tokenizer from pytorch_transformers
                - recreate: bool, True will force recreate db, default is False
                - score_limit: float, Limitation of score, if below this number will return None, default is 0.3
                - return_relate: bool, set to return related text, default is False
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

        self._build_index(file_path, recreate)
        self._load_model(model_path, self.device)
        self.score_limit = score_limit
        self.return_relate = return_relate

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

        stemmer = xapian.Stem("english")

        if not recreate:
            database = xapian.Database(cached_index)
        else:
            database = xapian.WritableDatabase(cached_index, xapian.DB_CREATE_OR_OPEN)
            indexer = xapian.TermGenerator()
            indexer.set_stemmer(stemmer)

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
                for l in tqdm(f, desc="Building index", unit=" lines"):
                    l = l.strip()
                    if len(l) < 1 :
                        if totS > 0 : totP += 1
                        totS = 0
                        continue
                    for sent in nltk.sent_tokenize(l):
                        sent.strip()
                        doc = xapian.Document()
                        doc.set_data(sent)
                        indexer.set_document(doc)
                        indexer.index_text(sent)
                        database.add_document(doc)

                        totN += 1
                        totS += 1
        
        self.parser = xapian.QueryParser()
        self.parser.set_stemmer(stemmer)
        self.parser.set_database(database)
        self.parser.set_stemming_strategy(xapian.QueryParser.STEM_SOME)
        self.enquire = xapian.Enquire(database)

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
        text = text.strip()
        if len(text) < 1:
            return None
        query = self.parser.parse_query(text)
        self.enquire.set_query(query)

        matches = self.enquire.get_mset(0, topN)
        return [str(m.document.get_data(), 'utf-8') for m in matches]

    def __call__(self, question, session_id=None):
        related_texts = self.search(question)
        _tmp = self.reader(question, "\n".join(related_texts))
        if _tmp is None:
            return None, 0
        example, feature, dataset = _tmp
        dataset = tuple(t.to(self.device) for t in dataset)
        outputs = self.model(input_ids = dataset[0],
                        attention_mask = dataset[1],
                        token_type_ids = dataset[2],
                        cls_index = dataset[4],
                        p_mask = dataset[5]
                       )
        
        answer, score = self.reader.convert_output_to_answer(example, feature, outputs, self.model.config.start_n_top, self.model.config.end_n_top)
        
        score = abs(score)
        
        text = answer["text"].strip()

        if len(text) < 1:
            return None, 0
        if score < self.score_limit:# or answer["probability"] < self.score_limit:
            print("text:", answer["text"], "probability:", answer["probability"], "score:", score)
            return None, 0

        if self.return_relate:
            text = answer["text"] + "\n\n" + "Related texts:\n* " + "\n* ".join(related_texts) + "\n"
        else:
            text = answer["text"]

        print("probability:", answer["probability"], "score:", score)
        return text, score

