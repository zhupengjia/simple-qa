#!/usr/bin/env python
import os, sqlite3, nltk, string, torch
from tqdm import tqdm
from nlptools.text import TFIDF
from nlptools.utils import lloads, ldumps
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
                 recreate=False):
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
        self.tokenizer = self.reader.tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        if not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        self.translator = str.maketrans('', '', string.punctuation) # remove punctuation
       
        self.tfidf = None
        self.cached_contents = file_path + ".db"
        self._build_db(file_path, recreate)
        self._build_index(file_path, recreate)
        self._load_model(model_path, self.device)

    def _get_cursor(self, dbfile):
        db = sqlite3.connect(dbfile)
        cursor = db.cursor()
        return db, cursor

    def _build_db(self, filepath, recreate=False):
        """
            save txt to LevelDB

            Input:
                - filepath: txt file path, support .gzip, .bzip2, and .txt file
                - recreate: bool, True will force recreate db, default is False
        """
        cached_contents = filepath + ".db"

        if os.path.exists(cached_contents):
            if recreate:
                os.remove(cached_contents)
        else:
            recreate = True

        db, cursor = self._get_cursor(cached_contents)
        if not recreate:
            totN = list(cursor.execute("select max(id) from contents"))[0][0]
            if totN is not None:
                db.close()
                return

        cursor.execute('''create table if not exists contents (
            id integer primary key,
            pid integer,
            sid integer,
            sentence blob,
            token_ids blob)''')

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
            batch_data = []
            for l in tqdm(f, desc="Converting to sqlite", unit=" lines"):
                l = l.strip()
                if len(l) < 1 :
                    if totS > 0 : totP += 1
                    totS = 0
                for sent in nltk.sent_tokenize(l):
                    sent = sent.strip()
                    clean_sent = sent.translate(self.translator).lower().strip()
                    if len(clean_sent) < 1:
                        continue
                    token_ids = self.tokenizer.encode(clean_sent)

                    if len(token_ids) < 1:
                        continue
                    batch_data.append((totN, totP, totS, ldumps(sent), ldumps(token_ids)))
                    totN += 1
                    totS += 1
                    if totN % 100000 == 0:
                        cursor.executemany("""insert into contents values (?,?,?,?,?)""", batch_data)
                        db.commit()
                        batch_data = []
            if len(batch_data) > 0:
                cursor.executemany("""insert into contents values (?,?,?,?,?)""", batch_data)
            db.commit()
        db.close()

    def _build_index(self, filepath, recreate=False):
        """
            build index

            Input:
                - filepath: txt file path, support .gzip, .bzip2, and .txt file
                - recreate: bool, True will force recreate db, default is False
        """
        cached_index = filepath + '.index'
        cached_contents = filepath + ".db"

        self.tfidf = TFIDF(vocab_size=self.vocab_size, cached_index=cached_index)

        db, cursor = self._get_cursor(cached_contents)
        totN = list(cursor.execute("select max(id) from contents"))[0][0] + 1
        db.close()

        def data_iter():
            db, cursor = self._get_cursor(cached_contents)
            for data in cursor.execute('select * from contents order by id'):
                yield lloads(data[4])
            db.close()
        self.tfidf.load_index(corpus_ids=data_iter(), corpus_len=totN, retrain=recreate)

    def _load_model(self, model_path, device):
        self.model = XLNetForQuestionAnswering.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def search(self, text, topN=5, before=0, after=0):
        """
            search related text

            Input:
                - text: string
                - topN: int, return topN results, default is 5
                - before: int, return Nlines before result, default is 0
                - after: int, return Nlines after result, default is 0
        """
        text = text.translate(self.translator).lower().strip()
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) < 1:
            return None
        result = self.tfidf.search_index(token_ids, topN=topN)

        db, cursor = self._get_cursor(self.cached_contents)
        sql = "select pid, sentence from contents where id=?"

        all_ids, all_sentences = [], []
        for i in [r[0] for r in result]:
            raw_pid, raw_sentence = list(cursor.execute(sql, (i,)))[0]

            ids, sentences = [], []
            for j in range(before, 0, -1):
                pid, sentence = list(cursor.execute(sql, (i-j,)))[0]
                if pid != raw_pid:
                    continue
                ids.append(i-j)
                sentences.append(sentence)
            ids.append(i)
            sentences.append(raw_sentence)
            for j in range(after):
                pid, sentence = list(cursor.execute(sql, (i+j+1,)))[0]
                if pid != raw_pid:
                    continue
                ids.append(i+j+1)
                sentences.append(sentence)
            all_ids += ids
            all_sentences += [lloads(s) for s in sentences]

        db.close()

        if len(all_ids) < 1:
            return None
        return all_sentences

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

