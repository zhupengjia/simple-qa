#!/usr/bin/env python
import json, os, torch
from pytorch_transformers.tokenization_bert import whitespace_tokenize
from pytorch_transformers import XLNetTokenizer
from torch.utils.data import TensorDataset
from .utils_squad import SquadExample, convert_examples_to_features

"""
    modified from pytorch_transformers project
"""

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
        self.is_training = is_training
        self.tokenizer = XLNetTokenizer.from_pretrained(tokenizer_name)

        cache_file = filename + "_{}_{}.features".format(tokenizer_name, max_seq_len)
        if os.path.exists(cache_file):
            self.examples, self.features = torch.load(cache_file)
        else:
            self.examples = self.read_example(filename)
            self.features = convert_examples_to_features(self.examples, self.tokenizer, max_seq_len, doc_stride, max_query_len, is_training)
            torch.save((self.examples, self.features), cache_file)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in self.features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self.features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in self.features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in self.features], dtype=torch.float)
        if is_training:
            all_start_positions = torch.tensor([f.start_position for f in self.features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in self.features], dtype=torch.long)
            self.dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask)
        else:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            self.dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_example_index, all_cls_index, all_p_mask)

    def convert_to_example(self,
                           qas_id,
                           question_text,
                           paragraph_text=None,
                           char_to_word_offset=None,
                           doc_tokens=None,
                           is_impossible=False,
                           answer=None,
                           answer_offset=None
                          ):
        """
            - qas_id: int
            - question_text: string
            - paragraph_text: string. If char_to_word_offset and doc_tokens exists, then you cal remain it to None
            - char_to_word_offset: list, it is the intermediate result after predealing the paragraph text
            - doc_tokens: list, it is the intermediate result after predealing the paragraph text
            - is_impossible: bool
            - answer: string
            - answer_offset: int, indicate the answer location in paragraph
        """
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        if char_to_word_offset is None or len(char_to_word_offset) < 1:
            if char_to_word_offset is None:
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

        start_position = None

        end_position = None
        orig_answer_text = None

        if self.is_training:
            if not is_impossible:
                orig_answer_text = answer
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
                    print("Could not find answer: '%s' vs. '%s'",
                                   actual_text, cleaned_answer_text)
                    return None
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ""
        else:
            is_impossible=False
    
        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        return example

    def read_example(self, input_file):
        """Read a SQuAD json file into a list of SquadExample."""
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
    
        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                doc_tokens = []
                char_to_word_offset = []
                for qa in paragraph["qas"]:
                    is_impossible = qa["is_impossible"]
                    example = self.convert_to_example(qas_id=qa["id"],
                                            question_text=qa["question"],
                                            paragraph_text=paragraph["context"],
                                            char_to_word_offset=char_to_word_offset,
                                            doc_tokens=doc_tokens,
                                            is_impossible=is_impossible,
                                            answer=qa["answers"][0]["text"] if not is_impossible else None,
                                            answer_offset=qa["answers"][0]["answer_start"] if not is_impossible else None
                                           )
                    if example:
                        examples.append(example)
        return examples

