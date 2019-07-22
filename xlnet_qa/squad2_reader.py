#!/usr/bin/env python
import json, collections, logging, os, torch
from pytorch_transformers.tokenization_bert import whitespace_tokenize
from pytorch_transformers import XLNetTokenizer

"""
    modified from pytorch_transformers project
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

class SQuAD2Reader:
    def __init__(self, filename, tokenizer_name="xlnet-base-cased", max_seq_len=384, doc_stride=128, max_query_len=64, is_training=True):
        """
            Reader for squad 2 dataset

            Input:
                - filename: string
                - tokenizer_name: string, default is xlnet_base_cased, tokenizer model name or path
                - max_seq_len: int, default is 384, The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.
                - doc_stride: int,  default is 128, When splitting up a long document into chunks, how much stride to take between chunks.
                - max_query_len: int, default is 64, The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
                - is_training: bool, default is True
        """
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride
        self.max_query_len = max_query_len
        self.is_training = is_training
        self.tokenizer = XLNetTokenizer.from_pretrained(tokenizer_name)
        cache_file = filename + ".features"
        if os.path.exists(cache_file):
            self.features = torch.load(cache_file)
        else:
            examples = self.read_example(filename)
            self.features = self.convert_features(examples)
            torch.save(self.features, cache_file)

    def read_example(self, input_file):
        """Read a SQuAD json file into a list of SquadExample."""
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
    
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False
    
        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)
    
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    if self.is_training:
                        is_impossible = qa["is_impossible"]
                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            raise ValueError(
                                "For training, each question should have exactly 1 answer.")
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                logger.warning("Could not find answer: '%s' vs. '%s'",
                                               actual_text, cleaned_answer_text)
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""
    
                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)
                    examples.append(example)
        return examples

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""
    
        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index
    
        return cur_span_index == best_span_index
    
    def _improve_answer_span(self, doc_tokens, input_start, input_end,
                             orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""
    
        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(self.tokenizer.tokenize(orig_answer_text))
    
        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)
    
        return (input_start, input_end)

    def convert_features(self, examples):
        
        """Loads a data file into a list of `InputBatch`s."""
        cls_token_at_end=False
        cls_token='[CLS]'
        sep_token='[SEP]'
        pad_token=0
        sequence_a_segment_id=0
        sequence_b_segment_id=1
        cls_token_segment_id=0
        pad_token_segment_id=0
        mask_padding_with_zero=True

        unique_id = 1000000000

        features = []
        for (example_index, example) in enumerate(examples):
            query_tokens = self.tokenizer.tokenize(example.question_text)

            if len(query_tokens) > self.max_query_len:
                query_tokens = query_tokens[0:self.max_query_len]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            if self.is_training and example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if self.is_training and not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position,
                    example.orig_answer_text)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = self.max_seq_len - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []

                # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
                # Original TF implem also keep the classification token (set to 0) (not sure why...)
                p_mask = []

                # CLS token at the beginning
                if not cls_token_at_end:
                    tokens.append(cls_token)
                    segment_ids.append(cls_token_segment_id)
                    p_mask.append(0)
                    cls_index = 0

                # Query
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(sequence_a_segment_id)
                    p_mask.append(1)

                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

                # Paragraph
                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(sequence_b_segment_id)
                    p_mask.append(0)
                paragraph_len = doc_span.length

                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(1)

                # CLS token at the end
                if cls_token_at_end:
                    tokens.append(cls_token)
                    segment_ids.append(cls_token_segment_id)
                    p_mask.append(0)
                    cls_index = len(tokens) - 1  # Index of classification token

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < self.max_seq_len:
                    input_ids.append(pad_token)
                    input_mask.append(0 if mask_padding_with_zero else 1)
                    segment_ids.append(pad_token_segment_id)
                    p_mask.append(1)

                assert len(input_ids) == self.max_seq_len
                assert len(input_mask) == self.max_seq_len
                assert len(segment_ids) == self.max_seq_len

                span_is_impossible = example.is_impossible
                start_position = None
                end_position = None
                if self.is_training and not span_is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                        span_is_impossible = True
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

                if self.is_training and span_is_impossible:
                    start_position = cls_index
                    end_position = cls_index

                if example_index < 20:
                    logger.info("*** Example ***")
                    logger.info("unique_id: %s" % (unique_id))
                    logger.info("example_index: %s" % (example_index))
                    logger.info("doc_span_index: %s" % (doc_span_index))
                    logger.info("tokens: %s" % " ".join(tokens))
                    logger.info("token_to_orig_map: %s" % " ".join([
                        "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                    logger.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                    ]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    logger.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    if self.is_training and span_is_impossible:
                        logger.info("impossible example")
                    if self.is_training and not span_is_impossible:
                        answer_text = " ".join(tokens[start_position:(end_position + 1)])
                        logger.info("start_position: %d" % (start_position))
                        logger.info("end_position: %d" % (end_position))
                        logger.info(
                            "answer: %s" % (answer_text))

                features.append(
                    InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        cls_index=cls_index,
                        p_mask=p_mask,
                        paragraph_len=paragraph_len,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=span_is_impossible))
                unique_id += 1

        return features


