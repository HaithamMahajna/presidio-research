from transformers import pipeline
from typing import List, Dict, Optional
from presidio_evaluator import InputSample, span_to_tag
from presidio_evaluator.models import BaseModel
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, BatchAnalyzerEngine

transformers_model = pipeline("token-classification", model="lakshyakh93/deberta_finetuned_pii")

class transformers_deberta_finetuned_pii(BaseModel) :
    def __init__(self,
                 entity_mapping: Optional[Dict[str, str]] = None,
        analyzer_engine: Optional[AnalyzerEngine] = None,
        entities_to_keep: List[str] = None,
        verbose: bool = False,
        labeling_scheme: str = "BIO",
        score_threshold: float = 0.4,
        language: str = "en",
        ad_hoc_recognizers: Optional[List[EntityRecognizer]] = None,
        context: Optional[List[str]] = None,
        allow_list: Optional[List[str]] = None,):
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
            entity_mapping=entity_mapping,
        )
        self.score_threshold = score_threshold
        self.language = language
        self.ad_hoc_recognizers = ad_hoc_recognizers
        self.context = context
        self.allow_list = allow_list

    def predict (self,sample) -> List[str]: 
        if type(sample)==str : sample = InputSample(full_text=sample)
        prediction = transformers_model(sample.full_text, aggregation_strategy="first")
        predictions = self.__recognizer_results_to_tags(prediction, sample)
        return predictions
    
    def batch_predict(self,dataset) -> List[List[str]]:
        predictions = []
        for data in dataset : 
            predictions.append(self.predict(data))
        return predictions


    
    

    @staticmethod
    def __recognizer_results_to_tags (result ,sample: InputSample
    ) -> List[str]:
        starts = []
        ends = []
        scores = []
        tags = []
        for res in result:
            starts.append(res['start'])
            ends.append(res['end'])
            tags.append(res['entity_group'])
            scores.append(res['score'])
        response_tags = span_to_tag(
            scheme="IO",
            text=sample.full_text,
            starts=starts,
            ends=ends,
            tokens=sample.tokens,
            scores=scores,
            tags=tags,
            )
        return response_tags
        



    

    
    

    
 


