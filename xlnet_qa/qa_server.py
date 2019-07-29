#!/usr/bin/env python
import os, sqlite3, spacy, multiprocessing
from tqdm import tqdm
from nlptools.text import TFIDF, Vocab
from nlptools.utils import zloads, zdumps
from pytorch_transformers import BertTokenizer
from spacy.tokens import Span

class QAServer:
    """
        QA restful server
    """
    def __init__(self, filepath, recreate=False, url_rule="/api/query", port=5002):
        """
            Input:
                - filepath: txt file path, support .gzip, .bzip2, and .txt file
                - tokenizer: tokenizer from pytorch_transformers
                - recreate: bool, True will force recreate db, default is False
        """
        self.tokenizer = spacy.load("en", disable=['tagger', 'ner', 'textcat', 'entity_ruler']) # get lemma and remove stopwords, punctuation
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True) #get sub words and related ids
        self.vocab_size = self.bert_tokenizer.vocab_size
       
        def bert_pipe(span):
            lemmas = [token.lemma_.lower() for token in span if not token.is_stop and not token.is_punct]
            if len(lemmas) < 1:
                return []
            return self.bert_tokenizer.encode(" ".join(lemmas))

        def bert_component(doc):
            doc.user_hooks["vector"] = bert_pipe
            doc.user_span_hooks["vector"] = bert_pipe
            return doc

        self.tokenizer.add_pipe(bert_component, name="bert", last=True)

        self.tfidf = None
        self.cached_contents = filepath + ".db"
        self._build_db(filepath, recreate)
        self._build_index(filepath, recreate)

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

        if not recreate:
            return

        db, cursor = self._get_cursor(cached_contents)

        cursor.execute('''create table contents (
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

        cpu_counts = max(multiprocessing.cpu_count()-2, 1)

        with open_func(filepath, mode="rt", encoding="utf-8") as f:
            totN, totP, totS= 0, 0, 0
            for doc in tqdm(self.tokenizer.pipe(f, n_threads=cpu_counts, batch_size=10000)):
                if not doc.text.strip():
                    if totS > 0 : totP += 1
                    totS = 0
                for sent in doc.sents:
                    if len(sent.vector) < 1:
                        continue

                    cursor.execute("""insert into contents values (?,?,?,?,?)""", (totN, totP, totS, zdumps(sent.text), zdumps(sent.vector)))
                    totN += 1
                    totS += 1
                    if totN % 100000 == 0:
                        db.commit()

            db.commit()
        db.close()

    def search(self, text, topN=5, before=0, after=0):
        """
            search related text

            Input:
                - text: string
                - topN: int, return topN results, default is 5
                - before: int, return Nlines before result, default is 0
                - after: int, return Nlines after result, default is 0
        """
        token_ids = self.tokenizer(text).vector
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
            all_sentences += [zloads(s) for s in sentences]

        db.close()

        if len(all_ids) < 1:
            return None
        return all_sentences

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
                yield zloads(data[4])
            db.close()
        self.tfidf.load_index(corpus_ids=data_iter(), corpus_len=totN, retrain=recreate)


