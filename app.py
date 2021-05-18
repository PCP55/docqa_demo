import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from collections import OrderedDict
import gradio as gr
from tika import parser

class DocumentQuestionAnswering:
  def __init__(self, pretrained_model_name_or_path='bert-large-uncased-whole-word-masking-finetuned-squad'):
    self.pretrained_model_name_or_path = pretrained_model_name_or_path
    self.device = [torch.cuda.current_device() if torch.cuda.is_available() else -1][0]
    self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
    self.model = AutoModelForQuestionAnswering.from_pretrained(self.pretrained_model_name_or_path).to(self.device)
    self.model_max_len = self.model.config.max_position_embeddings
    self.chunked = False

  def tokenize(self, context, question):
    self.inputs = self.tokenizer.encode_plus(question,
                                             context,
                                             add_special_tokens = True,
                                             return_tensors = 'pt')
    if 'token_type_ids' not in self.inputs.keys():
      self.has_token_type_ids = False
    else:
      self.has_token_type_ids = True

    if not self.has_token_type_ids:
      question_len = len(self.tokenizer.tokenize(question)) + 2 # Plus 2 means [CLS] x 1 and [SEP] x 1
      context_len = len(self.tokenizer.tokenize(context)) + 1
      question_token_type_ids = torch.tensor([0], requires_grad=False).repeat(question_len)
      context_token_type_ids = torch.tensor([1], requires_grad=False).repeat(context_len)
      token_type_ids = torch.cat((question_token_type_ids, context_token_type_ids)).unsqueeze(dim=0)
      self.inputs['token_type_ids'] = token_type_ids

    self.inputs = self.inputs.to(self.device)

    self.question_mask = self.inputs['token_type_ids'].lt(1)
    context_mask = torch.masked_select(self.inputs['input_ids'], ~self.question_mask)

    self.max_len = int((self.model_max_len - self.question_mask.size()[0])/2)

    if len(context_mask.tolist()) > self.max_len:
        self.inputs = self.chunkify()
        self.chunked = True

  def chunkify(self):
    chunked_input = OrderedDict()

    for k,v in self.inputs.items():
      question = torch.masked_select(v, self.question_mask)
      context = torch.masked_select(v, ~self.question_mask)
      chunks = torch.split(context, self.max_len)

      for i, chunk in enumerate(chunks):
        if i not in chunked_input:
          chunked_input[i] = {}
        thing = torch.cat((question, chunk))

        if i != len(chunks) - 1:
          if k == 'input_ids':
            thing = torch.cat((thing, torch.tensor([102]).to(self.device)))

          else:
            thing = torch.cat((thing, torch.tensor([1]).to(self.device)))
        chunked_input[i][k] = torch.unsqueeze(thing, dim = 0)
    return chunked_input

  def get_answer(self, top_n=5):
    answers_list = []
    scores_list = []
    start_list = []
    end_list = []
    sample_sentence_list = []

    if self.chunked:
      num_chunk = 0
      for _, chunk in self.inputs.items():
        if not self.has_token_type_ids:
          chunk.pop('token_type_ids')
        
        output = self.model(**chunk)
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits) + 1

        start_score = torch.max(output.start_logits)
        end_score = torch.max(output.end_logits)

        score = start_score + end_score
        ans = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(chunk['input_ids'][0][answer_start:answer_end]))

        if ('[CLS]' not in ans) and ('[SEP]' not in ans) and (ans.strip() != '<s>') and ((ans.strip() != '')):
          with torch.no_grad():
            answers_list.append(ans)
            scores_list.append(score.cpu())
            start_list.append((answer_start + (self.max_len * num_chunk)).cpu().numpy() + 0)
            end_list.append((((answer_end - 1) + (self.max_len * num_chunk))).cpu().numpy() + 0)
            sentence = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(chunk['input_ids'][0]))
            sep_pos = sentence.find('[SEP]')
            sentence = sentence[sep_pos+5:-5].strip()
            sample_sentence_list.append(sentence)
        num_chunk += 1

      if len(scores_list) > 0:
        answers = []
        scores_softmax = np.array(scores_list)
        exp_scores = np.exp(scores_softmax)
        probs_list = exp_scores / exp_scores.sum()

        for score, answer, start, end, sample_sentence in sorted(zip(probs_list, answers_list, start_list, end_list, sample_sentence_list), reverse=True):
          ans_dict = {}
          ans_dict['start'] = start
          ans_dict['end'] = end
          ans_dict['answer'] = answer
          ans_dict['score'] = score
          ans_dict['sample_sentence'] = sample_sentence
          answers.append(ans_dict)
        return answers[:top_n]
      else:
        return "Can't find answer"

    else:
      ans_dict = {}
      answers = []
      output = self.model(**self.inputs)
      answer_start = torch.argmax(output.start_logits)
      answer_end = torch.argmax(output.end_logits) + 1
      ans = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(self.inputs['input_ids'][0][answer_start:answer_end]))
      if ('[CLS]' not in ans) and ('[SEP]' not in ans) and (ans.strip() != '<s>') and ((ans.strip() != '')):
        ans_dict['start'] = answer_start.cpu().numpy() + 0
        ans_dict['end'] = answer_end.cpu().numpy() - 1
        ans_dict['answer'] = ans
        answers.append(ans_dict)
        return answers
      else:
        return "Can't find answer"

def clean_context(context: str) -> str:
  context = context.replace('\n',' ')
  context = context.replace('\t', ' ')
  context = ' '.join(context.split())

  return context

def file_and_question(file_obj, question):
  doc = parser.from_file(file_obj.name)
  context = doc["content"]
  context = clean_context(context)
  doc_qa = DocumentQuestionAnswering('bert-large-uncased-whole-word-masking-finetuned-squad')
  doc_qa.tokenize(context=context, question=question)

  answers = doc_qa.get_answer(top_n=3)
  rank_n_dict = {}
  for ans in answers:
    rank_n_dict[ans['answer']] = str(ans['score'])

  first_answer = answers[0]
  highlight_answer = []
  answer_pos = first_answer['sample_sentence'].find(first_answer['answer'])
  highlight_answer.append((first_answer['sample_sentence'][:answer_pos], 'Other'))
  highlight_answer.append((first_answer['sample_sentence'][answer_pos:answer_pos + len(first_answer['answer'])], 'Answer'))
  highlight_answer.append((first_answer['sample_sentence'][answer_pos + len(first_answer['answer']):], 'Other'))
  return highlight_answer, rank_n_dict

iface = gr.Interface(file_and_question, inputs=['file', 'text'], outputs=['highlight', 'key_values'])
# iface.test_launch()
iface.launch(debug=True)