'''
""" Official evaluation script for v1.1 of the SQuAD dataset. """
'''
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import numpy as np
import math
import operator

class JsonSerializable(object):
    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def __repr__(self):
        return self.toJson()

class Prediction(JsonSerializable):
    def __init__(self, text, probability):
        self.text = text
        self.probability = probability

class NBestPredictions(JsonSerializable):
    def __init__(self, qid, predictions):
        self.qid = qid
        self.predictions = []
        self.predictions = predictions
        

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def precision_score(prediction, ground_truth):
    
    prediction_set = set(prediction)
    ground_truth_set = set(ground_truth)
    
    intersection = prediction_set.intersection(ground_truth_set)
    
    if(len(intersection) == 0):
        return 0
    
    return len(intersection) / len(prediction)

def recall_score(prediction, ground_truth):
    
    prediction_set = set(prediction)
    ground_truth_set = set(ground_truth)
    
    intersection = prediction_set.intersection(ground_truth_set)
    
    if(len(intersection) == 0):
        return 0
    
    return len(intersection) / len(ground_truth)


def f1_score_list(prediction, ground_truth):
    
    precision = 1.0 * precision_score(prediction, ground_truth)
    if precision == 0:
        return 0, 0, 0
    
    recall = 1.0 * recall_score(prediction, ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def exact_sacc_score(nbest_prediction_text_first, exact_answers):
    """
    c1 is the number of factoid questions that have been answered correctly when only
    the first element of each returned list is considered
    """
    for answer in exact_answers:
        if (normalize_answer(nbest_prediction_text_first) == normalize_answer(answer)):
            return True
    
    
    return False

def exact_lacc_score(nbest_prediction_text, exact_answers):
    """
    c5 is the number of factoid questions that have
    been answered correctly in the lenient sense
    """
    prediction_order = 1
    for nbest_prediction in nbest_prediction_text:
        if prediction_order > 5:
            break
        
        for answer in exact_answers:
        
            if (normalize_answer(nbest_prediction) == normalize_answer(answer)):
                return True, prediction_order
        
        prediction_order += 1    

    return False, 0  

def precision_at_k(y_true, y_pred, k=10):
    """ Computes Precision at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Precision at k
    """
    intersection = np.intersect1d(y_true, y_pred[:k])
    return len(intersection) / k


def rel_at_k(y_true, y_pred, k=10):
    """ Computes Relevance at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Relevance at k
    """
    
    if(len(y_pred) < k):
        #print(str(len(y_pred)) + " : " + str(k) )
        return 0
    #else:
    #    print("noproblem:" + str(len(y_pred)) + " : " + str(k) )
        
    if y_pred[k-1] in y_true:
        return 1
    else:
        return 0
 
 
def average_precision_at_k(y_true, y_pred, k=10):
    """ Computes Average Precision at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Average Precision at k
    """
    ap = 0.0
    for i in range(1, k+1):
        ap += precision_at_k(y_true, y_pred, i) * rel_at_k(y_true, y_pred, i)
    
    #return ap
    
    #return ap / min(k, len(y_true)) 

    """
        average precision hesaplanırken toplam  average precision değeri relevant prediction sayısına bölünmeli
    """
    intersection = np.intersect1d(y_true, y_pred[:k]) # relevant prediction sayısı
    
    if(len(intersection) == 0):
        return 0
    
    return ap / min(k, len(intersection))  


def mean_average_precision(y_true, y_pred, k=10):
    """ Computes MAP at k
    
    Parameters
    __________
    y_true: np.array
            2D Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            2D Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           MAP at k
    """
    return np.mean([average_precision_at_k(gt, pred, k) \
                    for gt, pred in zip(y_true, y_pred)])
    
     
 
def compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs 


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def filter_question_by_id_fn(question_id, question):
    return question_id == question.id

def make_single(exact_answers):
    result = []
    
    for answers in exact_answers:
        if(isinstance(answers, list)):
            for answer in answers:
                result.append(answer)
        else:
            result.append(answers)
            
    return result

def evaluate(dataset, predictions, nbest_predictions, bioasq_golden_test, nbest_predictions_merge):
    f1 = exact_match = total = sacc = lacc = mrr = c1 = c5 = factoid_total = list_total = 0
    f1_factoid = precicion = recall = f1_list = precicion_list = recall_list =average_precision = map = 0.0
    qa_list = {}

    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                
                id = qa['id'].rpartition('_')[0]
                
                questions = list(filter(lambda question : (question['id'] == id),  bioasq_golden_test['questions']))
                
                
                if (len(questions) == 0):
                    continue
                
                
                if(questions[0]["type"] != "factoid" and questions[0]["type"] != "list"):
                    continue
                

                if(questions[0]["type"] == "factoid"):
                
                    factoid_total += 1
                                
                    exact_answers = make_single(questions[0]["exact_answer"])
            
                    nbest_prediction = nbest_predictions[qa['id']]
                    
                    nbest_prediction_text = []
                    for row in nbest_prediction:
                        nbest_prediction_text.append(row["text"])
            
                    #nbest_prediction_text = list(map(lambda x: x["text"], nbest_prediction))

                    if exact_sacc_score(nbest_prediction_text[0], exact_answers):
                        c1 += 1
                        c5 += 1
                        mrr += 1
                        f1_factoid += 1
                    else:
                        lenient_accuracy, prediction_order = exact_lacc_score(nbest_prediction_text, exact_answers)
                        
                        if(lenient_accuracy):
                            c5 += 1
                            mrr += 1 / prediction_order
                            
                            
                        f1, precicion, recall = f1_score_list(nbest_prediction_text[:5], exact_answers)
                        f1_factoid += f1
                            
                
                if(questions[0]["type"] == "list"):
                    """
                        https://stackoverflow.com/questions/40457331/information-retrieval-evaluation-python-precision-recall-f-score-ap-map
                        https://www.kaggle.com/code/debarshichanda/understanding-mean-average-precision
                    """
                    
                    #if id not in qa_list:
                    
                    list_total += 1
                    
                    exact_answers = make_single(questions[0]["exact_answer"])
            
                    nbest_prediction = nbest_predictions[qa['id']]
                    
                    nbest_prediction_text = []
                    for row in nbest_prediction:
                        nbest_prediction_text.append(row["text"])
                        
                    
                    nbest_prediction_merge = nbest_predictions_merge[id] 
                        
                    nbest_prediction_merge_text = []
                    for row in nbest_prediction_merge:
                        nbest_prediction_merge_text.append(row["text"])                        
                    
                    #nbest_prediction_text = list(map(lambda x: x['text'], nbest_prediction))
                    
                    #f1, precicion, recall = f1_score_list(nbest_prediction_merge_text[:5], exact_answers)
                    f1, precicion, recall = f1_score_list(nbest_prediction_merge_text[:len(exact_answers)*2], exact_answers)
                    f1_list += f1
                    precicion_list += precicion
                    recall_list += recall

                    average_precision += average_precision_at_k(exact_answers, nbest_prediction_text, 20)
                    
                ground_truths = []         
                for row in qa['answers']:
                    ground_truths.append(row["text"])
                    
                #ground_truths = list(map(lambda x: x['text'], qa['answers']))
                
                
                
                if id in qa_list:
                    qa_list[id] += ground_truths
                else:    
                    qa_list[id] = ground_truths
                
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    sacc = c1 / factoid_total
    lacc = c5 / factoid_total
    mrr = mrr / factoid_total
    f1_factoid = f1_factoid / factoid_total
    mean_average_precision = average_precision / list_total
    
    f1_list = f1_list / list_total
    precicion_list = precicion_list / list_total
    recall_list = recall_list / list_total
    

    return {'exact_match': exact_match, 'f1': f1, "SAcc" : sacc, "LAcc" : lacc, "MRR" : mrr, "total" : total, "factoid_total": factoid_total,
             "average_precision": average_precision, "MAP": mean_average_precision, "list_total": list_total, 
             "f1_list": f1_list, "precicion_list": precicion_list, "recall_list": recall_list,
             "f1_factoid" : f1_factoid}


def merge_prediction(dataset, predictions, nbest_predictions, bioasq_golden_test, nbest_prediction_file_name):
    f1 = exact_match = total = sacc = lacc = mrr = c1 = c5 = factoid_total = list_total = 0
    precicion = recall = f1_list = average_precision = map = 0.0
    qa_list = {}
    qa_pred_list = {}
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                
                id = qa['id'].rpartition('_')[0]
                
                questions = list(filter(lambda question : (question['id'] == id),  bioasq_golden_test['questions']))
                
                
                if (len(questions) == 0):
                    continue
                
                
                if(questions[0]["type"] != "factoid" and questions[0]["type"] != "list"):
                    continue
                

                if(questions[0]["type"] == "factoid"):
                
                    factoid_total += 1
                                
                    exact_answers = make_single(questions[0]["exact_answer"])
            
                    nbest_prediction = nbest_predictions[qa['id']]
                    
                    nbest_prediction_text = []
                    for row in nbest_prediction:
                        nbest_prediction_text.append(row["text"])
            
                    #nbest_prediction_text = list(map(lambda x: x["text"], nbest_prediction))

                    if exact_sacc_score(nbest_prediction_text[0], exact_answers):
                        c1 += 1
                        c5 += 1
                        mrr += 1
                    else:
                        lenient_accuracy, prediction_order = exact_lacc_score(nbest_prediction_text, exact_answers)
                        
                        if(lenient_accuracy):
                            c5 += 1
                            mrr += 1 / prediction_order
                
                if(questions[0]["type"] == "list"):
                    """
                        https://stackoverflow.com/questions/40457331/information-retrieval-evaluation-python-precision-recall-f-score-ap-map
                        https://www.kaggle.com/code/debarshichanda/understanding-mean-average-precision
                    """
                    
                    list_total += 1
                    
                    exact_answers = make_single(questions[0]["exact_answer"])
            
                    nbest_prediction = nbest_predictions[qa['id']]
                    
                    nbest_prediction_text = []
                    for row in nbest_prediction:
                        nbest_prediction_text.append(row["text"])
                        
                    nbest_prediction_probability = []
                    for row in nbest_prediction:
                        nbest_prediction_probability.append(row["probability"])
            
                    softmax_probability = compute_softmax(nbest_prediction_probability)
                    
                    if id in qa_pred_list:
                        qa_preds = qa_pred_list[id]
                        
                        row_index = 0
                        for row in nbest_prediction:
                            
                            found = False
                            for qa_pred in qa_preds:
                                if qa_pred["text"] == row["text"]:
                                    qa_pred["probability"] += softmax_probability[row_index]
                                    qa_pred["count"] += 1
                                    found = True
                                    
                            if not found:
                                pred = {}
                                pred["text"] = row["text"]
                                pred["probability"] = softmax_probability[row_index]
                                pred["count"] = 1
                                qa_preds.append(pred)
                            
                            row_index += 1
                    else:
                        
                        qa_pred_list[id] = []
                        qa_preds = qa_pred_list[id]
                        
                        row_index = 0
                        for row in nbest_prediction:
                            pred = {}
                            pred["text"] = row["text"]
                            pred["probability"] = softmax_probability[row_index]
                            pred["count"] = 1
                            qa_preds.append(pred)
                            
                            row_index += 1
                            
                            
                    #nbest_prediction_text = list(map(lambda x: x['text'], nbest_prediction))

                    average_precision += average_precision_at_k(exact_answers, nbest_prediction_text, 20)
                    
                ground_truths = []         
                for row in qa['answers']:
                    ground_truths.append(row["text"])
                    
                #ground_truths = list(map(lambda x: x['text'], qa['answers']))
                
                
                
                if id in qa_list:
                    qa_list[id] += ground_truths
                else:    
                    qa_list[id] = ground_truths
                
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    sacc = c1 / factoid_total
    lacc = c5 / factoid_total
    mrr = mrr / factoid_total
    mean_average_precision = average_precision / list_total


    #sorted_d = dict( sorted(qa_pred_list.items(), key=operator.itemgetter(1),reverse=True))
    #qa_pred_list.sort(key=lambda x: x[0]['probability'], reverse=True)
    
    """
    Sonuçlar probability alanına göre azalan şekilde sıralanıyor.
    https://www.programiz.com/python-programming/methods/list/sort
    """
    for qa_row in qa_pred_list:
        qa_pred_list[qa_row].sort(key=lambda x: x.get('probability'), reverse=True)

    json_string = json.dumps(qa_pred_list, default=lambda o: o.__dict__, indent=4)
    
    
    #jsonFile = open("test/bioasq9b-squadV1.json", "w")
    jsonFile = open(nbest_prediction_file_name + "_MERGE", "w")
    jsonFile.write( json_string )
    jsonFile.close() 

    return {'exact_match': exact_match, 'f1': f1, "SAcc" : sacc, "LAcc" : lacc, "MRR" : mrr, "total" : total, "factoid_total": factoid_total, "average_precision": average_precision, "MAP": mean_average_precision, "list_total": list_total}

def get_total_probability(text, json_file):
    total = 0.0
    
    for q in json_file:
        for row in json_file[q]:
            if row["text"] == text:
                total += row["probability"]
                print(row["text"] + str(row["probability"]))
    
    
    return total
        

def evaluate_orj(dataset, predictions, nbest_predictions, bioasq_golden_test):
    f1 = exact_match = total = 0
    qa_list = {}
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                
                id = qa['id'].rpartition('_')[0]
                
                if id in qa_list:
                    qa_list[id] += ground_truths
                else:    
                    qa_list[id] = ground_truths
                
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

if __name__ == '__main__':
    
    y_true = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    pred = np.array(['1', '0', '3', '4', '5', '6', '0', '0', '0', '10'])
    pred2 = np.array(['0', '2', '0', '0', '5', '6', '7', '0', '9', '10'])
    
    #print(average_precision_at_k(y_true, pred))
    #print(average_precision_at_k(y_true, pred2))
        
    
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    #squad test soruları dataset
    parser.add_argument('dataset_file', help='Dataset file')
    #prediction file
    parser.add_argument('prediction_file', help='Prediction File')
    #nbest prediction file
    parser.add_argument('nbest_prediction_file', help='NBest Prediction File')
    #bioasq golden test file
    parser.add_argument('bioasq_golden_test_file', help='Bioasq Golden Test File')
    
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
        
    with open(args.nbest_prediction_file) as nbest_prediction_file:
        nbest_predictions = json.load(nbest_prediction_file)
        
    with open(args.bioasq_golden_test_file) as bioasq_golden_test_file:
        bioasq_golden_test = json.load(bioasq_golden_test_file)
        
    with open(args.nbest_prediction_file + "_MERGE") as nbest_prediction_file_merge:
        nbest_predictions_merge = json.load(nbest_prediction_file_merge)        

    print(json.dumps(evaluate(dataset, predictions, nbest_predictions, bioasq_golden_test, nbest_predictions_merge)))
    
    #print(json.dumps(merge_prediction(dataset, predictions, nbest_predictions, bioasq_golden_test, args.nbest_prediction_file)))
    
