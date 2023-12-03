'''
Created on Feb 3, 2017

@author: harun
'''

import lucene
from org.apache.lucene.index import IndexReader
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.index import IndexWriter
from org.apache.lucene.index import IndexWriterConfig
from org.apache.lucene.index import Term;
#from org.apache.lucene.index.IndexWriterConfig import OpenMode
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.store import RAMDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search import PhraseQuery
from org.apache.lucene.search import TermQuery
from org.apache.lucene.search import BooleanQuery
#from org.apache.lucene.search.PhraseQuery import Builder
#from org.apache.lucene.search.BooleanQuery import Builder
from org.apache.lucene.search import BooleanClause;
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document
from org.apache.lucene.document import Field
from org.apache.lucene.document import StoredField
from org.apache.lucene.document import IntPoint
from org.apache.lucene.document import TextField
from org.apache.lucene.util import Version
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.queryparser.classic import QueryParserBase
from org.apache.lucene.queryparser.classic import MultiFieldQueryParser
from org.apache.lucene.search.similarities import LMDirichletSimilarity
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.search.similarities import ClassicSimilarity
from org.apache.lucene.search.similarities import LMJelinekMercerSimilarity

from java.io import File
from java.nio.file import Path
from java.nio.file import Paths
from pubmed.qanltk import stem_text
from nltk.util import ngrams
import pubmed.utils as utils
import lucene_qa.utils as qa_utils
import sys, traceback
import string
from nltk.tokenize import sent_tokenize
import copy


class PubmedArticle:
           
    def __init__(self, pmid, title, abstract, publicationDate, meshDescriptorUI, meshDescriptorConcept,  meshQualifierUI, 
                 meshQualifierConcept, chemicalUI, chemicalConcept, sentence_id,  score) :
        self.pmid = pmid
        self.title = title
        self.abstract = abstract
        self.publicationDate = publicationDate
        self.meshDescriptorUI = meshDescriptorUI
        self.meshDescriptorConcept = meshDescriptorConcept
        self.meshQualifierUI = meshQualifierUI
        self.meshQualifierConcept = meshQualifierConcept
        self.chemicalUI = chemicalUI
        self.chemicalConcept = chemicalConcept
        self.sentence_id =sentence_id
        self.score = score        


class ParsedSentenceSearchResult(object):
        
        def __init__(self, pmid, sentence, sentenceIndex, sentenceNer, sentenceCui, sentenceUmlsSemType, sentenceUmlsSemGroup, score):
            self._pmid = pmid
            self._sentence = sentence
            self._sentenceIndex = sentenceIndex
            self._sentenceNer = sentenceNer
            self._sentenceCui = sentenceCui
            self._sentenceUmlsSemType = sentenceUmlsSemType
            self._sentenceUmlsSemGroup = sentenceUmlsSemGroup
            self._score = score
             
        
        @property
        def pmid(self):
            return self._pmid    

        @property
        def sentence(self):
            return self._sentence    

        @property
        def sentenceIndex(self):
            return self._sentenceIndex    

        @property
        def sentenceNer(self):
            return self._sentenceNer    

        @property
        def sentenceCui(self):
            return self._sentenceCui    

        @property
        def sentenceUmlsSemType(self):
            return self._sentenceUmlsSemType    

        @property
        def sentenceUmlsSemGroup(self):
            return self._sentenceUmlsSemGroup   
         
        @property
        def score(self):
            return self._score   

        @score.setter
        def score(self, score):
            self._score = score  
            
        @property
        def transformer_answer(self):
            return self._transformer_answer   

        @transformer_answer.setter
        def transformer_answer(self, transformer_answer):
            self._transformer_answer = transformer_answer 
            
        @property
        def normalizationFactor(self):
            return self._normalization_factor   

        @normalizationFactor.setter
        def normalizationFactor(self, normalization_factor):
            self._normalization_factor = normalization_factor       

        @property
        def toString(self):
            return "pmid: {0:<15} | score: {1:<25} | {2}".format(str(self.pmid) + " - " + str(self.sentenceIndex), str(self.score), self.sentence)       
            
class ParsedSentenceSearchResultComp(object):
    
        def __init__(self, index, sentenceTextSimilarity, sentenceNerSimilarity, sentenceCuiSimilarity, sentenceUmlsSemTypeSimilarity, sentenceUmlsSemGroupSimilarity, parsedSentenceSearchResultScore):
            self._index = index
            self._sentenceTextSimilarity = sentenceTextSimilarity
            self._sentenceNerSimilarity = sentenceNerSimilarity
            self._sentenceCuiSimilarity = sentenceCuiSimilarity
            self._sentenceUmlsSemTypeSimilarity = sentenceUmlsSemTypeSimilarity
            self._sentenceUmlsSemGroupSimilarity = sentenceUmlsSemGroupSimilarity
            self._parsedSentenceSearchResultScore = parsedSentenceSearchResultScore

        @property
        def index(self):
            return self._index
        
        @property
        def sentenceTextSimilarity(self):
            return self._sentenceTextSimilarity
        
        @property
        def sentenceNerSimilarity(self):
            return self._sentenceNerSimilarity
        
        @property
        def sentenceCuiSimilarity(self):
            return self._sentenceCuiSimilarity
        
        @property
        def sentenceUmlsSemTypeSimilarity(self):
            return self._sentenceUmlsSemTypeSimilarity
        
        @property
        def sentenceUmlsSemGroupSimilarity(self):
            return self._sentenceUmlsSemGroupSimilarity
            
        @property
        def parsedSentenceSearchResultScore(self):
            return self._parsedSentenceSearchResultScore
        
        @property
        def toString(self):
            result = str(self.index) + " - "
            
            if(self.parsedSentenceSearchResultScore is not None):
                result += "\n{0:<30} : {1}".format("parsedSentenceSearchResult", self.parsedSentenceSearchResultScore.toString)

            if(self.sentenceTextSimilarity is not None):
                result += "\n{0:<30} : {1}".format("sentenceTextSimilarity", self.sentenceTextSimilarity.toString)
                 
            if(self.sentenceNerSimilarity is not None):
                result += "\n{0:<30} : {1}".format("sentenceNerSimilarity", self.sentenceNerSimilarity.toString) 

            if(self.sentenceCuiSimilarity is not None):
                result += "\n{0:<30} : {1}".format("sentenceCuiSimilarity", self.sentenceCuiSimilarity.toString) 

            if(self.sentenceUmlsSemTypeSimilarity is not None):
                result += "\n{0:<30} : {1}".format("sentenceUmlsSemTypeSimilarity", self.sentenceUmlsSemTypeSimilarity.toString) 

            if(self.sentenceUmlsSemGroupSimilarity is not None):
                result += "\n{0:<30} : {1}".format("sentenceUmlsSemGroupSimilarity", self.sentenceUmlsSemGroupSimilarity.toString) 


                
            
            return result    
        
class LuceneEngine:
    
    PMID_FIELD = "pmid"
    ARTICLE_TITLE_FIELD = "articleTitle"
    ABSTRACT_TEXT_FIELD = "abstractText"
    PUBLICATION_DATE = "publicationDate"
    
    MESH_DESCRITOR_UI = "meshDescriptorUI"
    MESH_DESCRITOR_CONCEPT = "meshDescriptorConcept"
    MESH_QUALIFIER_UI = "meshQualifierUI"
    MESH_QUALIFIER_CONCEPT = "meshQualifierConcept"

    CHEMICAL_UI = "chemicalUI"
    CHEMICAL_CONCEPT = "chemicalConcept"
    
    SENTENCE_TEXT = "sentence"
    SENTENCE_INDEX = "sentenceIndex"
    SENTENCE_NER = "sentenceNer"
    SENTENCE_CUI = "sentenceCui"
    SENTENCE_UMLS_SEM_TYPE = "sentenceUmlsSemType"
    SENTENCE_UMLS_SEM_GROUP = "sentenceUmlsSemGroup"


    def __init__(self) :
        self.pubmedArticleList = []
        self.ramPubmedArticleList = []
        jar_path = lucene.CLASSPATH # + ":/home/harun/pylucene-7.5.0/lucene-backward-codecs-7.5.0.jar"
        self.JCCEnv = lucene.initVM(jar_path)

        
    def get_similarity_class(self, similarity_name):
        if similarity_name == "tfIdfSimilarity":
            return ClassicSimilarity()
        if similarity_name == "bm25Similarity":
            return BM25Similarity()
        if similarity_name == "dirichletSimilarity":
            return LMDirichletSimilarity()
        if similarity_name == "jelinekMercerSimilarity":
            return LMJelinekMercerSimilarity(0.7)
        
        return LMDirichletSimilarity()
     

    def search(self, question, similarity_name, max_count):
        """
            Ref: https://lucene.apache.org/pylucene/features.html
            Before PyLucene APIs can be used from a thread other than the main thread that was not created by the Java Runtime, 
            the attachCurrentThread() method must be called on the JCCEnv object returned by the initVM() or getVMEnv() functions.
        """
        self.JCCEnv.attachCurrentThread()

        self.pubmedArticleList = []
        #print lucene.CLASSPATH
        #jarPath = lucene.CLASSPATH  + ":/home/harun/pylucene-6.2.0/lucene-backward-codecs-6.2.0.jar"
        #jarPath = lucene.CLASSPATH + ":/home/harun/pylucene-6.2.0/lucene-backward-codecs-7.2.0.jar"

        #jarPath = lucene.CLASSPATH + ":/home/harun/pylucene-7.5.0/lucene-backward-codecs-7.5.0.jar"
        #jarPath = lucene.CLASSPATH + ":lucene-backward-codecs-6.2.0.jar:/home/erdem/lucene-codecs-8.3.0.jar:/home/erdem/pylucene-8.3.0/lucene-backward-codecs-8.3.0.jar"
        
        #print("lucene_engine.search: " + str(jarPath))

        #lucene.initVM(jarPath)
        
        print(lucene.VERSION)

        analyzer = StandardAnalyzer()
       

        #indexDir = "/home/harun/medline_lucene_2017/"
        indexDir = "/home/erdem/medline_lucene_2019/"
        #indexDir = "/home/erdem/Lucene-allMeSH_2020"
        #indexDir = "/mnt/1816225D16223C5E/medline_lucene_2017"

        print(indexDir)

        path = Paths.get(indexDir);

        print(path);

        indexDirFS = SimpleFSDirectory(path)
    
    
        reader = DirectoryReader.open(indexDirFS);
        searcher = IndexSearcher(reader);
        
        #dirichletSimilarity = LMDirichletSimilarity();
        #searcher.setSimilarity(dirichletSimilarity);
        #bm25Similarity = BM25Similarity()
        #searcher.setSimilarity(bm25Similarity);
        
        searcher.setSimilarity(self.get_similarity_class(similarity_name));
    
        analyzer = StandardAnalyzer()
        #searcher = IndexSearcher(dir)
    
        fields = ["articleTitle", "abstractText", "meshDescriptorUI", "meshDescriptorConcept"]
#         question = "What symptoms characterize the Muenke syndrome?"
    
        parser = MultiFieldQueryParser(fields, analyzer)
        
        
        # parser.setDefaultOperator(QueryParserBase.AND_OPERATOR)
        
        #I think there's a bug with the method binding. MultiFieldQueryParser has several static parse methods, 
        # plus the inherited regular method from QueryParser. It looks like all of them are being resolved as if they were static. 
        # As a workaround, you can call it like this:               
        query = MultiFieldQueryParser.parse(parser, question)
        
        
        MAX = 20
        topDocs = searcher.search(query, max_count)
        print("Found %d document(s) that matched query '%s':" % (topDocs.totalHits.value, query))
        for  scoredoc in topDocs.scoreDocs:
            #print  scoredoc.ssudocore,  scoredoc.doc,  scoredoc.toString()
            doc = searcher.doc( scoredoc.doc)
            pmid = doc.get("pmid") #.encode("utf-8")
            title = doc.get("articleTitle") #.encode("utf-8")
            abstract = doc.get("abstractText") #.encode("utf-8")
            publicationDate = doc.get("publicationDate") #.encode("utf-8")
            meshDescriptorUI = doc.get("meshDescriptorUI") #.encode("utf-8")
            meshDescriptorConcept = doc.get("meshDescriptorConcept") #.encode("utf-8")
            
            meshQualifierUI = doc.get("meshQualifierUI") #.encode("utf-8")
            meshQualifierConcept = doc.get("meshQualifierConcept") #.encode("utf-8")
            
            chemicalUI = doc.get("chemicalUI") #.encode("utf-8")
            chemicalConcept = doc.get("chemicalConcept") #.encode("utf-8")
            
            
            #print "LuceneIndex pmid %s score '%s':" % (pmid,  scoredoc.score)
            #print doc.get("articleTitle") #.encode("utf-8")
            
            pubmed_article = PubmedArticle(pmid, title, abstract,  publicationDate, meshDescriptorUI, meshDescriptorConcept, 
                                           meshQualifierUI, meshQualifierConcept, chemicalUI, chemicalConcept, 0, scoredoc.score)
            self.pubmedArticleList.append(pubmed_article)
        
        reader.close()
        
        return topDocs, self.pubmedArticleList

    def searchByPMIDs(self, pmids, similarity_name, max_count):
        """
            Ref: https://lucene.apache.org/pylucene/features.html
            Before PyLucene APIs can be used from a thread other than the main thread that was not created by the Java Runtime, 
            the attachCurrentThread() method must be called on the JCCEnv object returned by the initVM() or getVMEnv() functions.
        """
        self.JCCEnv.attachCurrentThread()

        self.pubmedArticleList = []


        #indexDir = "/home/harun/medline_lucene_2017/"
        #indexDir = "/home/erdem/medline_lucene_2019/"
        
        indexDir = "/media/erdem/Windows8_OS/lucene_2022/"

        #print(indexDir)

        path = Paths.get(indexDir);

        #print(path);

        dir = SimpleFSDirectory(path)
    
    
        reader = DirectoryReader.open(dir);
        searcher = IndexSearcher(reader);
        searcher.setSimilarity(self.get_similarity_class(similarity_name));
        
        #doc = searcher.doc(pmids)
        #print(doc)
        #title = doc.get("pmid")
    
        #analyzer = StandardAnalyzer()
        #qp = QueryParser("pmid", analyzer)
        #idQuery = qp.parse(pmids)
        
        """
            Herhangi bir score hesaplanmıyor. pmid değeri ne ise ilgili makaleyi çeviriyor. Tek bir makale çeviriyor.
        """
        #query = IntPoint.newExactQuery("pmid", pmids)
        
        """
            Birden çok pmid değerine sahip list parametre olarak veriliyor.
        """
        query = IntPoint.newSetQuery("pmid", pmids)
        
        
        MAX = 20
        topDocs = searcher.search(query, max_count)
        #print("Found %d document(s) that matched query '%s'" % (topDocs.totalHits.value, query))
        
        for  scoredoc in topDocs.scoreDocs:
            #print  scoredoc.ssudocore,  scoredoc.doc,  scoredoc.toString()
            doc = searcher.doc( scoredoc.doc)
            pmid = doc.get("pmid") 
            title = doc.get("articleTitle")
            abstract = doc.get("abstractText")
            publicationDate = doc.get("publicationDate")
            meshDescriptorUI = doc.get("meshDescriptorUI")
            meshDescriptorConcept = doc.get("meshDescriptorConcept")
            
            meshQualifierUI = doc.get("meshQualifierUI")
            meshQualifierConcept = doc.get("meshQualifierConcept")
            
            chemicalUI = doc.get("chemicalUI")
            chemicalConcept = doc.get("chemicalConcept")
            
            
            #print "LuceneIndex pmid %s score '%s':" % (pmid,  scoredoc.score)
            #print doc.get("articleTitle") #.encode("utf-8")
            
            pubmed_article = PubmedArticle(pmid, title, abstract,  publicationDate, meshDescriptorUI, meshDescriptorConcept, 
                                           meshQualifierUI, meshQualifierConcept, chemicalUI, chemicalConcept, 0, scoredoc.score)
            self.pubmedArticleList.append(pubmed_article)
        
        reader.close()
        
        return topDocs, self.pubmedArticleList
        
    def search_trigram(self, question, max_count):
        self.pubmedArticleList = []
        #print lucene.CLASSPATH
        jarPath = lucene.CLASSPATH  + ":/home/harun/pylucene-6.2.0/lucene-backward-codecs-6.2.0.jar"
        #jarPath = lucene.CLASSPATH + ":/home/harun/pylucene-6.2.0/lucene-backward-codecs-7.2.0.jar"
        #print jarPath
        lucene.initVM(jarPath)
        indexDir = "/home/harun/medline_lucene_2017_three_sentence"
        path = Paths.get(indexDir);
        dir = SimpleFSDirectory(path)
    
    
        reader = DirectoryReader.open(dir);
        searcher = IndexSearcher(reader);
        
        dirichletSimilarity = LMDirichletSimilarity();
        searcher.setSimilarity(dirichletSimilarity);
        #bm25Similarity = BM25Similarity()
        #searcher.setSimilarity(bm25Similarity);
    
        analyzer = StandardAnalyzer()
        #searcher = IndexSearcher(dir)
    
        fields = ["articleTitle", "abstractText"]
    
        parser = MultiFieldQueryParser(fields, analyzer)
        
        
        # parser.setDefaultOperator(QueryParserBase.AND_OPERATOR)
        
        #I think there's a bug with the method binding. MultiFieldQueryParser has several static parse methods, 
        # plus the inherited regular method from QueryParser. It looks like all of them are being resolved as if they were static. 
        # As a workaround, you can call it like this:               
        query = MultiFieldQueryParser.parse(parser, question)
        
        
        MAX = 20
        topDocs = searcher.search(query, max_count)
        #print "Found %d document(s) that matched query '%s':" % (topDocs.totalHits.value, query)
        for  scoredoc in topDocs.scoreDocs:
            #print  scoredoc.score,  scoredoc.doc,  scoredoc.toString()
            doc = searcher.doc( scoredoc.doc)
            pmid = doc.get("pmid") #.encode("utf-8")
            title = doc.get("articleTitle") #.encode("utf-8")
            abstract = doc.get("abstractText") #.encode("utf-8")
            sentence_id = doc.get("sid") #.encode("utf-8")
            publicationDate = None
            
            meshDescriptorUI = None
            meshDescriptorConcept = None
            
            meshQualifierUI = None
            meshQualifierConcept = None
            
            chemicalUI = None
            chemicalConcept = None

            
            #print "LuceneIndex pmid %s score '%s':" % (pmid,  scoredoc.score)
            #print doc.get("articleTitle") #.encode("utf-8")
            
            pubmed_article = PubmedArticle(pmid, title, abstract,  publicationDate, meshDescriptorUI, meshDescriptorConcept, 
                                           meshQualifierUI, meshQualifierConcept, chemicalUI, 
                                           chemicalConcept, sentence_id, scoredoc.score)
            self.pubmedArticleList.append(pubmed_article)
        
        reader.close()
        
        return topDocs, self.pubmedArticleList
        
        
    def writeAndSearchRAMIndex(self, question, similarity_name,  pubmedArticleList, max_count, top_n):
        #Create RAMDirectory instance
        ramDir = RAMDirectory()
         
        #Builds an analyzer with the default stop words
        analyzer = StandardAnalyzer()
         
        #Write some docs to RAMDirectory
        self.writeRAMIndex(ramDir, analyzer, pubmedArticleList)
        
        #Search indexed docs in RAMDirectory
        pubmedArticleListUniqram = self.searchRAMIndex(ramDir, analyzer, question, similarity_name, max_count);   


        ramDir = RAMDirectory()

        #Write some docs to RAMDirectory
        self.writeSentenceRAMIndex(ramDir, analyzer, pubmedArticleList)

        
         
        #Search indexed docs in RAMDirectory
        pubmedArticleListSentence = self.searchRAMIndex(ramDir, analyzer, question, similarity_name, max_count);   
        
        
        for pubmedArticle in pubmedArticleListSentence:                  
            for unigramPubmedArticle in pubmedArticleListUniqram:
                if(unigramPubmedArticle.pmid == pubmedArticle.pmid):
                    pubmedArticle.score = (pubmedArticle.score * 0.65) + (unigramPubmedArticle.score * 0.35)
                    break
   
        
        
        pubmedArticleListSentence.sort(key=lambda pubmedArticle: pubmedArticle.score, reverse=True)
        
        topN = []
        for pubmedArticle in pubmedArticleListSentence:
            
            if(len(topN) == top_n):
                break
            
            already_added = False
            for pubmedArticleAdded in topN:
                if(pubmedArticleAdded.pmid == pubmedArticle.pmid):
                    already_added = True
                    break
            
            if not already_added:
                topN.append(pubmedArticle)
            
       
        
        if True:
            #topN = pubmedArticleListSentence[0:30]
            return topN
        
        
        bigram = utils.to_ngrams(question, n=2)
        
        #print bigram
        
        
        queryBuilder = BooleanQuery.Builder()
                       
        
        for grams in bigram:
            #title
            phraseQueryTitleBuilder = PhraseQuery.Builder();
            phraseQueryTitleBuilder.setSlop(10);
            phraseQueryTitleBuilder.add(Term("articleTitle", grams[0]));
            phraseQueryTitleBuilder.add(Term("articleTitle", grams[1]));
            phraseQueryTitle = phraseQueryTitleBuilder.build();
            queryBuilder.add(phraseQueryTitle,  BooleanClause.Occur.SHOULD);
            
            #abstract
            phraseQueryAbstractBuilder = PhraseQuery.Builder();
            phraseQueryAbstractBuilder.setSlop(10);
            phraseQueryAbstractBuilder.add(Term("abstractText", grams[0]));
            phraseQueryAbstractBuilder.add(Term("abstractText", grams[1]));
            phraseQueryAbstract = phraseQueryAbstractBuilder.build();
            queryBuilder.add(phraseQueryAbstract,  BooleanClause.Occur.SHOULD);
            
            #meshDescriptorUI
            phraseQueryAbstractBuilder = PhraseQuery.Builder();
            phraseQueryAbstractBuilder.setSlop(10);
            phraseQueryAbstractBuilder.add(Term("meshDescriptorUI", grams[0]));
            phraseQueryAbstractBuilder.add(Term("meshDescriptorUI", grams[1]));
            phraseQueryAbstract = phraseQueryAbstractBuilder.build();
            queryBuilder.add(phraseQueryAbstract,  BooleanClause.Occur.SHOULD);
        
        
        pubmedArticleListBiqram = self.searchRAMIndexPhaseQuery(ramDir, analyzer, queryBuilder.build(), similarity_name,  max_count);
        

        trigram = utils.to_ngrams(question, n=3)
        
        #print trigram
        
        
        queryBuilder = BooleanQuery.Builder()
                       
        
        for grams in trigram:
            #title
            phraseQueryTitleBuilder = PhraseQuery.Builder();
            phraseQueryTitleBuilder.setSlop(10);
            phraseQueryTitleBuilder.add(Term("articleTitle", grams[0]));
            phraseQueryTitleBuilder.add(Term("articleTitle", grams[1]));
            phraseQueryTitleBuilder.add(Term("articleTitle", grams[2]));
            phraseQueryTitle = phraseQueryTitleBuilder.build();
            queryBuilder.add(phraseQueryTitle,  BooleanClause.Occur.SHOULD);
            
            #abstract
            phraseQueryAbstractBuilder = PhraseQuery.Builder();
            phraseQueryAbstractBuilder.setSlop(10);
            phraseQueryAbstractBuilder.add(Term("abstractText", grams[0]));
            phraseQueryAbstractBuilder.add(Term("abstractText", grams[1]));
            phraseQueryAbstractBuilder.add(Term("abstractText", grams[2]));
            phraseQueryAbstract = phraseQueryAbstractBuilder.build();
            queryBuilder.add(phraseQueryAbstract,  BooleanClause.Occur.SHOULD);
        
            #meshDescriptorUI
            phraseQueryAbstractBuilder = PhraseQuery.Builder();
            phraseQueryAbstractBuilder.setSlop(10);
            phraseQueryAbstractBuilder.add(Term("meshDescriptorUI", grams[0]));
            phraseQueryAbstractBuilder.add(Term("meshDescriptorUI", grams[1]));
            phraseQueryAbstractBuilder.add(Term("meshDescriptorUI", grams[2]));
            phraseQueryAbstract = phraseQueryAbstractBuilder.build();
            queryBuilder.add(phraseQueryAbstract,  BooleanClause.Occur.SHOULD);
        
        pubmedArticleListTriqram = self.searchRAMIndexPhaseQuery(ramDir, analyzer, queryBuilder.build(), similarity_name, max_count);
        
        
        pubmedArticleList = []
        
        for pubmedArticle in pubmedArticleListBiqram:                  
            for unigramPubmedArticle in pubmedArticleListUniqram:
                if(unigramPubmedArticle.pmid == pubmedArticle.pmid):
                    #unigramPubmedArticle.score += (pubmedArticle.score * 1.30)
                    unigramPubmedArticle.score += (unigramPubmedArticle.score * 1.50)
                    
        for pubmedArticle in pubmedArticleListTriqram:                  
            for unigramPubmedArticle in pubmedArticleListUniqram:
                if(unigramPubmedArticle.pmid == pubmedArticle.pmid):
                    #unigramPubmedArticle.score += (pubmedArticle.score * 1.50)
                    unigramPubmedArticle.score += (unigramPubmedArticle.score * 2.50)
            
        
        
        pubmedArticleListUniqram.sort(cmp=None,  key=lambda pubmedArticle: pubmedArticle.score, reverse=True)
        
        #for unigramPubmedArticle in pubmedArticleListUniqram:
        #    print "RamIndex pmid %s score '%s':" % (unigramPubmedArticle.pmid,  unigramPubmedArticle.score)
        
        topN = pubmedArticleListUniqram[0:20]
        
        #for unigramPubmedArticle in topN:
        #    print "RamIndex pmid %s score '%s':" % (unigramPubmedArticle.pmid,  unigramPubmedArticle.score)
             
        
        return topN
        
     
    def writeRAMIndex(self, ramDir, analyzer, pubmedArticleList):
        try:
            #IndexWriter Configuration
            iwc = IndexWriterConfig(analyzer)
            iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
 
            #IndexWriter writes new index files to the directory
            writer = IndexWriter(ramDir, iwc)
            
            for pubmedArticle in pubmedArticleList:
                doc = Document()
                doc.add(IntPoint(self.PMID_FIELD, int(pubmedArticle.pmid)));
                doc.add(StoredField(self.PMID_FIELD, int(pubmedArticle.pmid)));
                doc.add(TextField(self.ARTICLE_TITLE_FIELD, pubmedArticle.title, Field.Store.YES));
                doc.add(TextField(self.ABSTRACT_TEXT_FIELD, pubmedArticle.abstract, Field.Store.YES));
                #doc.add(TextField(self.ARTICLE_TITLE_FIELD, stem_text(pubmedArticle.title), Field.Store.YES));
                #doc.add(TextField(self.ABSTRACT_TEXT_FIELD, stem_text(pubmedArticle.abstract), Field.Store.YES));

                doc.add(TextField(self.PUBLICATION_DATE, pubmedArticle.publicationDate, Field.Store.YES));
                
                doc.add(TextField(self.MESH_DESCRITOR_UI, pubmedArticle.meshDescriptorUI, Field.Store.YES));
                doc.add(TextField(self.MESH_DESCRITOR_CONCEPT, pubmedArticle.meshDescriptorConcept, Field.Store.YES));
                
                doc.add(TextField(self.MESH_QUALIFIER_UI, pubmedArticle.meshQualifierUI, Field.Store.YES));
                doc.add(TextField(self.MESH_QUALIFIER_CONCEPT, pubmedArticle.meshQualifierConcept, Field.Store.YES));
                
                doc.add(TextField(self.CHEMICAL_UI, pubmedArticle.chemicalUI, Field.Store.YES));
                doc.add(TextField(self.CHEMICAL_CONCEPT, pubmedArticle.chemicalConcept, Field.Store.YES));
                
                writer.addDocument(doc);
            
            writer.commit()
            writer.close()
        
        except:
            #Any error goes here
            #e.printStackTrace();
            print("Exception in user code:")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            writer.close()


    def writeSentenceRAMIndex(self, ramDir, analyzer, pubmedArticleList):
        try:
            #IndexWriter Configuration
            iwc = IndexWriterConfig(analyzer)
            iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
 
            #IndexWriter writes new index files to the directory
            writer = IndexWriter(ramDir, iwc)
            
            for pubmedArticle in pubmedArticleList:
                
                sentence_group_list = utils.sentence_slide_window(pubmedArticle.abstract, 3)
                
                for sentence_group in sentence_group_list:
                
                    sentence_doc = " ".join(sentence_group)
                    
                    doc = Document()
                    doc.add(IntPoint(self.PMID_FIELD, int(pubmedArticle.pmid)));
                    doc.add(StoredField(self.PMID_FIELD, int(pubmedArticle.pmid)));
                    doc.add(TextField(self.ARTICLE_TITLE_FIELD, pubmedArticle.title, Field.Store.YES));
                    doc.add(TextField(self.ABSTRACT_TEXT_FIELD, sentence_doc, Field.Store.YES));
                    #doc.add(TextField(self.ARTICLE_TITLE_FIELD, stem_text(pubmedArticle.title), Field.Store.YES));
                    #doc.add(TextField(self.ABSTRACT_TEXT_FIELD, stem_text(pubmedArticle.abstract), Field.Store.YES));
    
                    doc.add(TextField(self.PUBLICATION_DATE, pubmedArticle.publicationDate, Field.Store.YES));
                    
                    doc.add(TextField(self.MESH_DESCRITOR_UI, pubmedArticle.meshDescriptorUI, Field.Store.YES));
                    doc.add(TextField(self.MESH_DESCRITOR_CONCEPT, pubmedArticle.meshDescriptorConcept, Field.Store.YES));
                    
                    doc.add(TextField(self.MESH_QUALIFIER_UI, pubmedArticle.meshQualifierUI, Field.Store.YES));
                    doc.add(TextField(self.MESH_QUALIFIER_CONCEPT, pubmedArticle.meshQualifierConcept, Field.Store.YES));
                    
                    doc.add(TextField(self.CHEMICAL_UI, pubmedArticle.chemicalUI, Field.Store.YES));
                    doc.add(TextField(self.CHEMICAL_CONCEPT, pubmedArticle.chemicalConcept, Field.Store.YES));
                    
                    writer.addDocument(doc);
            
            writer.commit()
            writer.close()
        
        except:
            #Any error goes here
            #e.printStackTrace();
            print("Exception in user code:")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            writer.close()
        
    def searchRAMIndex(self, ramDir, analyzer, question, similarity_name, max_count):
        self.ramPubmedArticleList = []
        try:
            #question = stem_text(question)
            #Create Reader
            reader = DirectoryReader.open(ramDir);
             
            #Create index searcher
            searcher = IndexSearcher(reader);
            #dirichletSimilarity = LMDirichletSimilarity(5.0);
            #searcher.setSimilarity(dirichletSimilarity);
            
            #bm25Similarity = BM25Similarity()
            #searcher.setSimilarity(bm25Similarity);
            
            searcher.setSimilarity(self.get_similarity_class(similarity_name));

            
            #Build query
            fields = ["articleTitle", "abstractText", "meshDescriptorUI", "meshDescriptorConcept"]
            parser = MultiFieldQueryParser(fields, analyzer)
            query = MultiFieldQueryParser.parse(parser, question)
            
            print(query)
            
            #Search the index
            topDocs = searcher.search(query, max_count)
            
            print("Found %d document(s) that matched query '%s':" % (topDocs.totalHits.value, query))
            
            for scoredoc in topDocs.scoreDocs:
                #print  scoredoc.score,  scoredoc.doc,  scoredoc.toString()
                doc = searcher.doc(scoredoc.doc)
                pmid = doc.get("pmid") #.encode("utf-8")
                title = doc.get("articleTitle") #.encode("utf-8")
                abstract = doc.get("abstractText") #.encode("utf-8")
                publicationDate = doc.get("publicationDate") #.encode("utf-8")
                meshDescriptorUI = doc.get("meshDescriptorUI") #.encode("utf-8")
                meshDescriptorConcept = doc.get("meshDescriptorConcept") #.encode("utf-8")
                
                meshQualifierUI = doc.get("meshQualifierUI") #.encode("utf-8")
                meshQualifierConcept = doc.get("meshQualifierConcept") #.encode("utf-8")
                
                chemicalUI = doc.get("chemicalUI") #.encode("utf-8")
                chemicalConcept = doc.get("chemicalConcept") #.encode("utf-8")

                #print "RamIndex pmid %s score '%s':" % (pmid,  scoredoc.score)
                #print doc.get("articleTitle") #.encode("utf-8")
                
                pubmed_article = PubmedArticle(pmid, title, abstract,  publicationDate, meshDescriptorUI, meshDescriptorConcept,
                                               meshQualifierUI, meshQualifierConcept, chemicalUI, chemicalConcept, 0, scoredoc.score)
                self.ramPubmedArticleList.append(pubmed_article)

            reader.close()
            
            return self.ramPubmedArticleList
        except:
            #
            print("Exception in user code:")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            reader.close()
            
    def searchRAMIndexPhaseQuery(self, ramDir, analyzer, query, similarity_name, max_count):
        print(query)
        
        pubmedArticleList = []
        try:
            #question = stem_text(question)
            #Create Reader
            reader = DirectoryReader.open(ramDir);
             
            #Create index searcher
            searcher = IndexSearcher(reader);
            #dirichletSimilarity = LMDirichletSimilarity(5.0);
            #searcher.setSimilarity(dirichletSimilarity);
            
            searcher.setSimilarity(self.get_similarity_class(similarity_name));

            
            #Build query
            #fields = ["articleTitle", "abstractText", "meshDescriptorUI", "meshDescriptorConcept"]
            #parser = MultiFieldQueryParser(fields, analyzer)
            #query = MultiFieldQueryParser.parse(parser, question)
            
            #Search the index
            topDocs = searcher.search(query, max_count)
            
            print("Found %d document(s) that matched query '%s':" % (topDocs.totalHits.value, query))
            
            for scoredoc in topDocs.scoreDocs:
                #print  scoredoc.score,  scoredoc.doc,  scoredoc.toString()
                doc = searcher.doc(scoredoc.doc)
                pmid = doc.get("pmid") #.encode("utf-8")
                title = doc.get("articleTitle") #.encode("utf-8")
                abstract = doc.get("abstractText") #.encode("utf-8")
                publicationDate = doc.get("publicationDate") #.encode("utf-8")
                meshDescriptorUI = doc.get("meshDescriptorUI") #.encode("utf-8")
                meshDescriptorConcept = doc.get("meshDescriptorConcept") #.encode("utf-8")
                
                meshQualifierUI = doc.get("meshQualifierUI") #.encode("utf-8")
                meshQualifierConcept = doc.get("meshQualifierConcept") #.encode("utf-8")
                
                chemicalUI = doc.get("chemicalUI") #.encode("utf-8")
                chemicalConcept = doc.get("chemicalConcept") #.encode("utf-8")

                #print "phaseQuery pmid %s score '%s':" % (pmid,  scoredoc.score)
                #print doc.get("articleTitle") #.encode("utf-8")
                
                pubmed_article = PubmedArticle(pmid, title, abstract,  publicationDate, meshDescriptorUI, meshDescriptorConcept,
                                               meshQualifierUI, meshQualifierConcept, chemicalUI, chemicalConcept, 0, scoredoc.score)
                pubmedArticleList.append(pubmed_article)

            reader.close()
            
            return pubmedArticleList
        except:
            #
            print("Exception in user code:")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            reader.close()
            


    """
        parsedSentence search
    """       
    
    def writeAndSearchParsedSentenceToRAMIndex(self, questionParsedSentence, similarity_name,  parsedSentenceList, max_count, top_n, sentence_only):
        import datetime
        
        #Create RAMDirectory instance
        ramDir = RAMDirectory()
         
        #Builds an analyzer with the default stop words
        analyzer = StandardAnalyzer()
         
        #Write parsedSentenceList to RAMDirectory
        self.writeParsedSentenceToRAMIndex(ramDir, analyzer, parsedSentenceList)
        
        #Search indexed docs in RAMDirectory
        sentenceTextSimilarityList = self.searchParsedSentenceRAMIndex([self.SENTENCE_TEXT], ramDir, analyzer, questionParsedSentence.sentence.replace("/", " ").replace("?",""), similarity_name, max_count);
        
        if (sentence_only):
            return sentenceTextSimilarityList
           
        sentenceNerSimilarityList = self.searchParsedSentenceRAMIndex([self.SENTENCE_NER], ramDir, analyzer, questionParsedSentence.snerSentence.replace(",", ""), similarity_name, max_count);
        sentenceCuiSimilarityList = self.searchParsedSentenceRAMIndex([self.SENTENCE_CUI], ramDir, analyzer, questionParsedSentence.cuiSentence.replace(",", ""), similarity_name, max_count);
        sentenceUmlsSemTypeSimilarityList = self.searchParsedSentenceRAMIndex([self.SENTENCE_UMLS_SEM_TYPE], ramDir, analyzer, questionParsedSentence.umlsSemTypeSentence.replace(",", ""), similarity_name, max_count);
        sentenceUmlsSemGroupSimilarityList = self.searchParsedSentenceRAMIndex([self.SENTENCE_UMLS_SEM_GROUP], ramDir, analyzer, questionParsedSentence.umlsGroupSemTypeSentence.replace(",", ""), similarity_name, max_count);


        """
         Tüm sonuçların score değerleri [0,1] aralığına normalize ediliyor.   
        """
        qa_utils.normalizeScore(sentenceTextSimilarityList)
        qa_utils.normalizeScore(sentenceNerSimilarityList)
        qa_utils.normalizeScore(sentenceCuiSimilarityList)
        qa_utils.normalizeScore(sentenceUmlsSemTypeSimilarityList)
        qa_utils.normalizeScore(sentenceUmlsSemGroupSimilarityList)
        
        """ """
        text_weight = 0.2311
        ner_weight = 0.1552
        cui_weight = 0.2387
        semtype_weight = 0.2068
        semgroup_weight = 0.1682
        
        
        """
        text_weight = 0.3528
        cui_weight = 0.3416
        semtype_weight = 0.3056
        """
        
        """
        text_weight = 0.44637382
        ner_weight = -0.00326242
        cui_weight = 0.37772929
        semtype_weight = 0.24057619
        semgroup_weight = 0.11617712
        """
        
        """
        text_weight = 0.2
        ner_weight = 0.2
        cui_weight = 0.2
        semtype_weight = 0.2
        semgroup_weight = 0.2
        """
        
        parsedSentenceSearchResultScoreList = []
        
        self.calculateParsedSentenceScore(parsedSentenceSearchResultScoreList, sentenceTextSimilarityList, text_weight)
        self.calculateParsedSentenceScore(parsedSentenceSearchResultScoreList, sentenceNerSimilarityList, ner_weight)
        self.calculateParsedSentenceScore(parsedSentenceSearchResultScoreList, sentenceCuiSimilarityList, cui_weight)
        self.calculateParsedSentenceScore(parsedSentenceSearchResultScoreList, sentenceUmlsSemTypeSimilarityList, semtype_weight)
        self.calculateParsedSentenceScore(parsedSentenceSearchResultScoreList, sentenceUmlsSemGroupSimilarityList, semgroup_weight)
        
        
        parsedSentenceSearchResultScoreList.sort(key=lambda parsedSentenceSearchResult: parsedSentenceSearchResult.score, reverse=True)
        
        #for parsedSentenceSearchResultScore in parsedSentenceSearchResultScoreList:
        #  print("RamIndex pmid %s %s, score '%s', sentence: %s" , (parsedSentenceSearchResultScore.pmid,  parsedSentenceSearchResultScore.score, parsedSentenceSearchResultScore.sentence))
        
        
        parsedSentenceSearchResultCompList = []
        
        for i in range(200):
            sentenceTextSimilarity = qa_utils.getFromListByIndex(sentenceTextSimilarityList, i)
            sentenceNerSimilarity = qa_utils.getFromListByIndex(sentenceNerSimilarityList, i)
            sentenceCuiSimilarity = qa_utils.getFromListByIndex(sentenceCuiSimilarityList, i)
            sentenceUmlsSemTypeSimilarity = qa_utils.getFromListByIndex(sentenceUmlsSemTypeSimilarityList, i)
            sentenceUmlsSemGroupSimilarity = qa_utils.getFromListByIndex(sentenceUmlsSemGroupSimilarityList, i)
            parsedSentenceSearchResultScore = qa_utils.getFromListByIndex(parsedSentenceSearchResultScoreList, i)
            
        
            parsedSentenceSearchResultComp = ParsedSentenceSearchResultComp(i, sentenceTextSimilarity, sentenceNerSimilarity, sentenceCuiSimilarity, sentenceUmlsSemTypeSimilarity, sentenceUmlsSemGroupSimilarity, parsedSentenceSearchResultScore)
            
            parsedSentenceSearchResultCompList.append(parsedSentenceSearchResultComp)
            
        """
        uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.') + ".sim"
        with open("./result/" + uniq_filename , 'a') as out:
            out.write("Query: " + questionParsedSentence.sentence + "\n")
            for ps in parsedSentenceSearchResultCompList:
                out.write(ps.toString + "\n")
        
        """
        
        #topN = parsedSentenceSearchResultScoreList[0:50]
        
        #return topN
        
        return parsedSentenceSearchResultScoreList, sentenceTextSimilarityList, sentenceNerSimilarityList, sentenceCuiSimilarityList, sentenceUmlsSemTypeSimilarityList, sentenceUmlsSemGroupSimilarityList
        

    def writeAndSearchAllParsedSentenceToRAMIndex(self, questionParsedSentence, similarity_name,  parsedSentenceList, max_count, top_n):
        import datetime
        
        #Create RAMDirectory instance
        ramDir = RAMDirectory()
         
        #Builds an analyzer with the default stop words
        analyzer = StandardAnalyzer()
         
        #Write parsedSentenceList to RAMDirectory
        self.writeParsedSentenceToRAMIndex(ramDir, analyzer, parsedSentenceList)
        
        #Search indexed docs in RAMDirectory
        search_fields = [self.SENTENCE_TEXT, self.SENTENCE_NER, self.SENTENCE_CUI, self.SENTENCE_UMLS_SEM_TYPE, self.SENTENCE_UMLS_SEM_GROUP]
        query = questionParsedSentence.sentence.replace("/", " ").replace("?","") + " " + questionParsedSentence.snerSentence.replace(",", "") + " " + questionParsedSentence.cuiSentence.replace(",", "") + " " + questionParsedSentence.umlsSemTypeSentence.replace(",", "") + " " + questionParsedSentence.umlsGroupSemTypeSentence.replace("," , "")
        
        parsedSentenceSearchResultList = self.searchParsedSentenceRAMIndex(search_fields, ramDir, analyzer, query, similarity_name, max_count);
      
        
        #parsedSentenceSearchResultScoreList.sort(key=lambda parsedSentenceSearchResult: parsedSentenceSearchResult.score, reverse=True)
        
        #for parsedSentenceSearchResultScore in parsedSentenceSearchResultList:
        #    print("RamIndex pmid %s %s, score '%s', sentence: %s" , (parsedSentenceSearchResultScore.pmid,  parsedSentenceSearchResultScore.score, parsedSentenceSearchResultScore.sentence))
        
        
       
        #topN = parsedSentenceSearchResultScoreList[0:50]
        
        #return topN
        
        return parsedSentenceSearchResultList
        
        
        

    def writeParsedSentenceToRAMIndex(self, ramDir, analyzer, parsedSentenceList):
        try:
            #IndexWriter Configuration
            iwc = IndexWriterConfig(analyzer)
            iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
 
            #IndexWriter writes new index files to the directory
            writer = IndexWriter(ramDir, iwc)
            
            
            for parsedSentence in parsedSentenceList:
                doc = Document()
                doc.add(IntPoint(self.PMID_FIELD, int(parsedSentence.pmid)));
                doc.add(StoredField(self.PMID_FIELD, int(parsedSentence.pmid)));
                doc.add(TextField(self.SENTENCE_TEXT, parsedSentence.sentence.replace("/"," ").replace("?",""), Field.Store.YES));
                doc.add(StoredField(self.SENTENCE_INDEX, int(parsedSentence.sentenceIndex)));
                
                #doc.add(TextField(self.SENTENCE_NER, parsedSentence.snerSentence, Field.Store.YES));
                #doc.add(TextField(self.SENTENCE_CUI, parsedSentence.cuiSentence, Field.Store.YES));
                #doc.add(TextField(self.SENTENCE_UMLS_SEM_TYPE, parsedSentence.umlsSemTypeSentence, Field.Store.YES));
                #doc.add(TextField(self.SENTENCE_UMLS_SEM_GROUP, parsedSentence.umlsGroupSemTypeSentence, Field.Store.YES));
                
                doc.add(TextField(self.SENTENCE_NER, parsedSentence.snerSentence.replace(",", ""), Field.Store.YES));
                doc.add(TextField(self.SENTENCE_CUI, parsedSentence.cuiSentence.replace(",", ""), Field.Store.YES));
                doc.add(TextField(self.SENTENCE_UMLS_SEM_TYPE, parsedSentence.umlsSemTypeSentence.replace(",", ""), Field.Store.YES));
                doc.add(TextField(self.SENTENCE_UMLS_SEM_GROUP, parsedSentence.umlsGroupSemTypeSentence.replace("," , ""), Field.Store.YES));

                writer.addDocument(doc);
            
            writer.commit()
            writer.close()
        
        except:
            #Any error goes here
            #e.printStackTrace();
            print("Exception in user code:")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            writer.close()
            
    def searchParsedSentenceRAMIndex(self, fields, ramDir, analyzer, question, similarity_name, max_count):
        parsedSentenceSearchResultList = []
        try:
            #question = stem_text(question)
            #Create Reader
            reader = DirectoryReader.open(ramDir);
             
            #Create index searcher
            searcher = IndexSearcher(reader);
            #dirichletSimilarity = LMDirichletSimilarity(5.0);
            #searcher.setSimilarity(dirichletSimilarity);
            
            #bm25Similarity = BM25Similarity()
            #searcher.setSimilarity(bm25Similarity);
            
            searcher.setSimilarity(self.get_similarity_class(similarity_name));

            
            #Build query
            #fields = ["articleTitle", "abstractText", "meshDescriptorUI", "meshDescriptorConcept"]
            parser = MultiFieldQueryParser(fields, analyzer)
            query = MultiFieldQueryParser.parse(parser, question)
            
            print(query)
            
            #Search the index
            topDocs = searcher.search(query, max_count)
            
            print("Found %d document(s) that matched query '%s':" % (topDocs.totalHits.value, query))
            
            for scoredoc in topDocs.scoreDocs:
                #print(scoredoc.score,  scoredoc.doc,  scoredoc.toString(), sep=",")
                doc = searcher.doc(scoredoc.doc)
                pmid = doc.get(self.PMID_FIELD)
                sentence = doc.get(self.SENTENCE_TEXT)
                sentenceIndex = doc.get(self.SENTENCE_INDEX)
                sentenceNer = doc.get(self.SENTENCE_NER)
                sentenceCui = doc.get(self.SENTENCE_CUI)
                sentenceUmlsSemType = doc.get(self.SENTENCE_UMLS_SEM_TYPE)
                sentenceUmlsSemGroup = doc.get(self.SENTENCE_UMLS_SEM_GROUP)
    

                
                parsedSentenceSearchResult = ParsedSentenceSearchResult(pmid, sentence, sentenceIndex, sentenceNer, sentenceCui, sentenceUmlsSemType, sentenceUmlsSemGroup, scoredoc.score)
                parsedSentenceSearchResultList.append(parsedSentenceSearchResult)

            reader.close()
            
            return parsedSentenceSearchResultList
        
        except:
            #
            print("Exception in user code:")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            reader.close()            
    


    
    def calculateParsedSentenceScore(self, parsedSentenceSearchResultScoreList, parsedSentenceSearchResultList, weight = 0.2):
        
        if (parsedSentenceSearchResultList is None or len(parsedSentenceSearchResultList) == 0):
            return;
        
        for parsedSentenceSearchResult in parsedSentenceSearchResultList:
            
            found = False
            for i in range(len(parsedSentenceSearchResultScoreList)):
                
                scoreSentence = parsedSentenceSearchResultScoreList[i]
                
                #Cümle, daha önce scoreList eklenmişse, score güncelleniyor.
                if (scoreSentence.pmid == parsedSentenceSearchResult.pmid and scoreSentence.sentenceIndex == parsedSentenceSearchResult.sentenceIndex):
                    scoreSentence.score += parsedSentenceSearchResult.score * weight
                    
                    parsedSentenceSearchResultScoreList[i] = scoreSentence
                    found = True
                    break
            
            #Cümle daha önce listeye hiç eklenmemişse listeye ekleniyor.
            if not found:
                #yeni bir obje oluşturuluyor ve score değeri değiştiriliyor.
                temp = copy.deepcopy(parsedSentenceSearchResult)
                temp.score = temp.score * weight
                parsedSentenceSearchResultScoreList.append(temp)
                #print(temp.toString)
                #print(parsedSentenceSearchResult.toString)
            
                
            
            #if utils.contains(scoreList, lambda x: x.pmid == parsedSentenceSearchResult.pmid and x.sentenceIndex = parsedSentenceSearchResult.sentenceIndex)
                
            
            
       
            


    #def buildPhaseQuery(self):
        #preferred_name_array_list = utils.listToTwoDimensionList(preferred_name_list)
        #PhraseQuery.Builder builder = PhraseQuery.Builder()
        #builder.add(Term("articleTitle", "SYNDROME"), 1)
        #builder.add(Term("articleTitle", "MUENKE"), 2)
        #builder.setSlop(10)
        #PhraseQuery phraseQuery = builder.build()
        
        #return phraseQuery
        

#lucene_engine = LuceneEngine()

if __name__ == "__main__":
    
    import sys
    
    print(sys.version)
    

    lucene_engine = LuceneEngine()
    print('lucene', lucene.VERSION)

     #lucene_engine = LuceneEngine()
    question = 'What symptoms characterize the Muenke syndrome?'
    question = "D004724"
    max_count = 20
    #lucene_engine.search(question, "dirichletSimilarity", max_count)
    #lucene_engine.searchByPMIDs(question,"9624027 18976161 8824885", "dirichletSimilarity", max_count)
    #lucene_engine.searchByPMIDs(question,"9624027", "dirichletSimilarity", max_count)
    pmids = [9624027,18976161, 8824885]
    lucene_engine.searchByPMIDs(pmids, "dirichletSimilarity", max_count)
    

    #question = "What symptoms characterize the Muenke syndrome?"
    #lucene_engine.search(question, "dirichletSimilarity", max_count)
    #lucene_engine.search(question, "dirichletSimilarity", max_count)
        
