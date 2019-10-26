#!/usr/bin/env python
import json, os, torch, collections
from transformers.tokenization_bert import whitespace_tokenize
from transformers import DistilBertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import SequentialSampler, DataLoader
from .utils_squad import SquadExample, convert_examples_to_features, get_final_text, _compute_softmax, RawResult, _get_best_indexes

"""
    modified from pytorch_transformers project
"""

def to_list(tensor):
    return tensor.detach().cpu().tolist()

class SQuAD2Reader:
    def __init__(self, tokenizer_name="distilbert-base-uncased-distilled-squad", max_seq_len=384, doc_stride=128, max_query_len=64, is_training=True):
        """
            Reader for squad 2 dataset

            Input:
                - tokenizer_name: string, default is xlnet_base_cased, tokenizer model name or path
                - max_seq_len: int, default is 384, The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.
                - doc_stride: int,  default is 128, When splitting up a long document into chunks, how much stride to take between chunks.
                - max_query_len: int, default is 64, The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
                - is_training: bool, default is True
        """
        self.is_training = is_training
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride
        self.max_query_len = max_query_len
        self.tokenizer_name = tokenizer_name

    def squad_data(self, filename):
        """
            predeal squad dataset
        """
        cache_file = filename + "_{}_{}.features".format(self.tokenizer_name, self.max_seq_len)
        if os.path.exists(cache_file):
            examples, features = torch.load(cache_file)
        else:
            examples = self.read_example(filename)
            features = convert_examples_to_features(examples, self.tokenizer, self.max_seq_len, self.doc_stride, self.max_query_len, self.is_training)
            torch.save((examples, features), cache_file)

        # Convert to Tensors and build dataset
        datasets = self.features_to_datasets(features)
        return examples, features, datasets

    def features_to_datasets(self, features):
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        if self.is_training:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask)
        else:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_example_index, all_cls_index, all_p_mask)
        return dataset

    def convert_to_example(self,
                           question_text,
                           qas_id=None,
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

    def convert_output_to_answer(self, example, feature, model_output, n_best_size=20, max_answer_length=30):
        """Convert model's output to answer
            Requires utils_squad_evaluate.py
        """
        unique_id = int(feature.unique_id)
        result = RawResult(unique_id    = unique_id,
                           start_logits = to_list(model_output[0][0]),
                           end_logits   = to_list(model_output[1][0]))

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["start_index", "end_index", "start_logit", "end_logit"])
        
        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])
        
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        start_indexes = _get_best_indexes(result.start_logits, n_best_size)
        end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            
        # if we could have irrelevant answers, get the min score of irrelevant
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
            score_null = feature_null_score
            null_start_logit = result.start_logits[0]
            null_end_logit = result.end_logits[0]

        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index]))
        prelim_predictions.append(
            _PrelimPrediction(
                start_index=0,
                end_index=0,
                start_logit=null_start_logit,
                end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)


        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, True, False)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if "" not in seen_predictions:
            nbest.append(
                _NbestPrediction(
                    text="",
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
            
        # In very rare edge cases we could only have single null prediction.
        # So we just create a nonce prediction in this case to avoid failure.
        if len(nbest)==1:
            nbest.insert(0,
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        
        score_diff = score_null
        #return  best_non_null_entry, score_diff
        return nbest_json[0], score_diff

    def __call__(self, question, paragraph):
        """ get feature for eval"""
        example = self.convert_to_example(
            question_text = question,
            paragraph_text = paragraph
        )
        feature = convert_examples_to_features([example], self.tokenizer, self.max_seq_len, self.doc_stride, self.max_query_len, False)
        if len(feature) < 1:
            return None
        dataset = self.features_to_datasets(feature)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)
        return example, feature[0], next(iter(dataloader))

