

import codecs
import json
import datetime
from types import SimpleNamespace
from json import JSONEncoder

from nltk import word_tokenize
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize


from difflib import SequenceMatcher

import  pubmed.utils as utils
from builtins import isinstance

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID
from typing import List
from boto.rds import logfile
#from dacite import from_dict



class MyJSONEncoder(JSONEncoder):
    def default(self, obj):
        return obj.to_json()
   
    

class JsonSerializable(object):
    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def __repr__(self):
        return self.toJson()


class SquadDataSetAnswer(JsonSerializable):
       
    def __init__(self, text, answer_start):
        self.text = text
        self.answer_start = answer_start

 
    """ 
    @property
    def text(self):
        return self.text   
   
    @text.setter
    def text(self, text):
        self.text = text  
            
    
    @property
    def answer_start(self):
        return self._answer_start   
  
    @answer_start.setter
    def answer_start(self, answer_start):
        self._answer_start = answer_start   
    
    #@property
    #def toString(self):
    #    return "pmid: {0:<15} | score: {1:<25} | {2}".format(str(self.pmid) + " - " + str(self.sentenceIndex), str(self.score), self.sentence)       
       
    """

class SquadDataSetQas(JsonSerializable):
    
    def __init__(self, id, question, answer):
        self.id = id
        self.question = question
        self.answers = []
        self.answers.append(answer)

    """
    @property
    def id(self):
        return self._id
    
   
    @id.setter
    def text(self, id):
        self._id = id  
            
    
    @property
    def question(self):
        return self._question   
    
    @question.setter
    def question(self, question):
        self._question = question   
     
    @property
    def answers(self):
        return self._answers  
    
    @answers.setter
    def answers(self, answers):
        self._answers = answers   

    """
class SquadDataSetParagraph(JsonSerializable):
    
    def __init__(self, context, qas):
        self.context = context
        self.qas = []
        self.qas.append(qas)

    """
    @property
    def context(self):
        return self._context
    
   
    @context.setter
    def context(self, context):
        self._context = context  
            
    
    @property
    def qas(self):
        return self._qas   
    
    @qas.setter
    def qas(self, qas):
        self._qas = qas   
    """

class Prediction(JsonSerializable):
    def __init__(self, text, probability):
        self.text = text
        self.probability = probability

class NBestPredictions(JsonSerializable):
    def __init__(self, qid, predictions):
        self.qid = qid
        self.predictions = []
        self.predictions = predictions


@dataclass
class TimeWindow:
    free_capacity: bool
    capacity_template_identifier: str
    cut_off: str
    end: str
    max_capacity: int
    start: str
    time_window_identifier: str
    tsp_id: int

def get_similar_ratio(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def make_single(exact_answers):
    result = []
    
    for answers in exact_answers:
        if(isinstance(answers, list)):
            for answer in answers:
                result.append(answer)
        else:
            result.append(answers)
            
    return result

def to_ngrams(text, n=2):
    token = word_tokenize(text)
    return ngrams(token, n)


def find_similar_context(answer, context, similarity_trashold = 1):
    answer_len = len(answer.split())
    
    ngrams = to_ngrams(context, answer_len)
    #similarity_trashold = 0.85
    max_similarity_ratio = 0.0
    
    selected_answer = None
    for n in ngrams:
        row = " ".join(n)
        
        similarity_ratio = get_similar_ratio(answer, row)
        
        if(similarity_ratio >= similarity_trashold and similarity_ratio > max_similarity_ratio):
            max_similarity_ratio = similarity_ratio
            selected_answer = row 
    
    
    return selected_answer
    

def find_answer_start_index(answer, context):
    """context te cevap yer alıyor mu bakılıyor"""
    answer_lower = answer.lower()
    context_lower = context.lower()
    
    answer_len = len(answer.split())
    
    """ tek kelimelik cevaplar için tokenize edilerek benzerlik karşılaştırılıyor, diğer durumda ayrı bir kelimde içermeyen 
        benzer metin içeren ifadelerde seçilebiliyor."""
    if(answer_len == 1):
        similar_context = find_similar_context(answer, context, similarity_trashold = 1)
        
        if(similar_context is not None):
            answer_start_index = context.find(similar_context)
            
        else:
            answer_start_index = -1 
    else: 
        answer_start_index = context_lower.find(answer_lower)
    
    if(answer_start_index >= 0):
        """küçük harfe dönüştürülerek karşılaştırma yapıldığı için orjinal context'ten ilgili kısım cevap olarak alınıyor"""
        end_position = answer_start_index + len(answer)
        actual_text = context[answer_start_index:end_position]
        
        if actual_text.find(answer) == -1:
            print("Could not find answer: " + actual_text + " vs. " +  answer)
        else:
            print("Found answer: " + actual_text + " vs. " +  answer)    
        
        return answer_start_index, actual_text.strip()
    
    """ en az 4 karakter ise benzerliğe bakılıyor """
    if(len(answer) >= 4):
        """ context yer almıyorsa (farklı yazım veya karakter farkı olabilir), benzerlikten bakılıyor."""
        similar_context = find_similar_context(answer, context)
        
        if(similar_context is not None):
            
            answer_start_index = context.find(similar_context)
            
            print("similar_context answer found. A/S " + answer + " vs. " +  similar_context)
            
            return answer_start_index, similar_context
        else:
            print("similar_context answer no found. A:" + answer)

    return -1, None

if __name__ == "__main__":
    
    from lucene_qa.retrieval import LuceneEngine # lucene_engine
    lucene_engine = LuceneEngine()

    p_file_type = "test" # test veya train olabilir.
    paragraphs = []
    
    #train_file_path = "test/training10b.json"
    
    #train_file_path = "test/BioASQ-training9b/training9b.json"
    
    train_file_path = "/home/erdem/Task9BGoldenEnriched/9B5_golden.json"
    
    with codecs.open(train_file_path, 'r', 'utf-8') as data_file:    
        data = json.load(data_file)
    
    d_data = json.dumps(data)    
    
    train_data = json.loads(d_data, object_hook=lambda d: SimpleNamespace(**d))
    
    #ortak_soru_sayisi = 0
    omitted_q_count = 0
    row_index = 0
    factoid_q = list_q = yesno_q = summary_q = 0
    for row in train_data.questions:
        row_index = row_index + 1;
        
        #if row.id != "601c17c21cb411341a00000e":
        #    continue
        
        
        """
        soru tiplerini saymak için
        """
        
        if(row.type == "yesno"):
            yesno_q += 1
        elif row.type == "summary":
            summary_q += 1
        elif row.type == "list":
            list_q += 1
        elif row.type == "factoid":
            factoid_q += 1
            
        #continue    
        
        
        """ test soruları train datata yer alıyor mu kontrolü """
        """
        test_file_path = "./test/Task10BTestSet/BioASQ-task10bPhaseB-testset6.json"
        with open(test_file_path) as bioasq_golden_test_file:
            bioasq_golden_test = json.load(bioasq_golden_test_file)

        
        
        
        questions = list(filter(lambda question : (question['id'] == row.id),  bioasq_golden_test['questions']))
                
        if (len(questions) > 0):
            ortak_soru_sayisi += 1
            print(str(ortak_soru_sayisi) + " - " + str(row_index) + " - " + row.body + " - " + row.id)
        
            
        
        
        if True:
            continue   
        """     
        
        """ test soruları train datata yer alıyor mu kontrolü sonu """
        
        
        if(row.type == "yesno" or row.type == "summary"):
            continue 
        
        print(str(row_index) + " - " + row.body)
        
        #if(row.body != "Which computational methods are used for the definition of synteny?"):
        #    continue
        
        """
        "list"
        "yesno"
        "factoid"
        "summary"
        """
        
        ideal_answer = row.ideal_answer
        exact_answers = make_single(row.exact_answer)
        
        snippet_count = 1
        for snippet in row.snippets:
            #print(snippet.text)
            #print(snippet.offsetInBeginSection)
            #print(snippet.document)   
            
            pmid = snippet.document.rsplit("/",1)[-1]   
            
            
            """ lucene den makalenin abstract kısmı çekiliyor. """
            pmidList = [int(pmid.strip())]
            hits, pubmedArticleList = lucene_engine.searchByPMIDs(pmidList, "dirichletSimilarity", 1)
            
            
            if(len(pubmedArticleList) == 0):
                continue
            
            """ Cevap içeren cümle title ise makalenin title'ı context olarak alınıyor. Diğer durumda abstract alınıyor. """
            
            if (snippet.beginSection == "title"):
                context = pubmedArticleList[0].title
            else:
                context = pubmedArticleList[0].abstract      
        
            """ answer alınıyor"""
            
            selected_answer = None
            answer_found_for_snippet = False
            for answer in exact_answers:
                """ test soru seti oluştururken her bir snippet için tek bir soru ekleniyor. """
                if p_file_type == "test" and answer_found_for_snippet:
                    omitted_q_count += 1
                    continue 
                
                """context te cevap aranıyor"""
                answer_start_index, selected_answer = find_answer_start_index(answer, context)
                
                """ context cevap bulunamamışsa ve en az bir sefer ilgili makalede cevap bulunamamışsa ideal answer'a bakılıyor."""
                #if(answer_start_index == -1 and not answer_found_for_snippet):
                #    """ Benzer içerik te bulunamazsa- context olarak ideal_answer alınıyor ve cevap bulunmaya çalışılıyor"""
                #    context = context +  " " + ideal_answer[0]
                #    answer_start_index, selected_answer = find_answer_start_index(answer, context)
                    
                """cevap metinde yer almıyor"""    
                if(answer_start_index == -1 or answer is None):
                    continue
                
                answer_found_for_snippet = True
                squadDataSetAnswer = SquadDataSetAnswer(selected_answer, answer_start_index)
                #squadDataSetAnswer = SquadDataSetAnswer(snippet.text, snippet.offsetInBeginSection)
                
                #squadData = json.dumps(squadDataSetAnswer.__dict__)
                
                """qas part oluşturuluyor"""
                squadDataSetQas = SquadDataSetQas(row.id + "_" + str(snippet_count).zfill(3), row.body, squadDataSetAnswer)
                
                #json_data = json.dumps(squadDataSetQas.__dict__, default=lambda o: o.__dict__, indent=4)
            
            
                    
                """ paragraph oluşturuluyor """
                squadDataSetParagraph = SquadDataSetParagraph(context, squadDataSetQas)
                
                
                #json_data = json.dumps(squadDataSetParagraph.__dict__, default=lambda o: o.__dict__, indent=4)
                
                paragraphs.append(squadDataSetParagraph)
                
                
                snippet_count = snippet_count + 1
            
            
            if not answer_found_for_snippet:
                logfile = open("test/9B5_golden_testV6_log.json", "a")
                logfile.write("row.id: " + row.id + " pmid: " + pmid + "\r\n")
                logfile.close()
        
        #squadData = json.dumps(paragraphs)  
        #json_string = json.dumps([ob.__dict__ for ob in paragraphs])
        #json_string = json.dumps(paragraphs, default=lambda o: o.__dict__, indent=4)
        
        #jsonFile = open("test/squadDataExport.json", "w")
        #jsonFile.write(json_string)
        #jsonFile.close()  
        
        #break
        
        #for doc in row.documents:
        #    print(doc)
            
        #for answer in row.ideal_answer:
            #print(answer)
            
        #print(row.type)
         
        #print(row.id)
         
        
    
    print("factoid:" + str(factoid_q) + " list:" + str(list_q) + " yesno:" + str(yesno_q) + " summary:" + str(summary_q))
    
    print("Omitted_q_count: " + str(omitted_q_count))
    
    json_string = json.dumps(paragraphs, default=lambda o: o.__dict__, indent=4)
    
    
    #jsonFile = open("test/bioasq9b-squadV1.json", "w")
    jsonFile = open("test/9B5_golden_testV6--4.json", "w")
    jsonFile.write( "{\n\"data\": [\n{\n    \"paragraphs\":" + json_string + ",\n \"title\": \"BioASQ9b\" \n         }\n      ],\"version\": \"BioASQ9b\" \n}")
    jsonFile.close()  

    #print(x.name, x.hometown.name, x.hometown.id)
               
        
                    