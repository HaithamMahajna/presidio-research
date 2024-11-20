from transformers import pipeline
from typing import List, Dict, Optional
from presidio_evaluator import InputSample, span_to_tag
from presidio_evaluator.models import BaseModel
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, BatchAnalyzerEngine



class transformers_model(BaseModel) :
    def __init__(self,
        entity_mapping: Optional[Dict[str, str]] = None,
        entities_to_keep: List[str] = None,
        verbose: bool = False,
        labeling_scheme: str = "BIO",
        model_name: str = None,
        aggregation_strategy:str = None,):
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
            entity_mapping=entity_mapping,
        )
        self.model_name = model_name
        self.aggregation_strategy = aggregation_strategy
        if self.model_name : self.transformers_model = pipeline("token-classification", model=self.model_name)
        if not self.model_name : self.transformers_model = pipeline("token-classification", model="lakshyakh93/deberta_finetuned_pii")
        

    def predict (self,sample) -> List[str]: 
        if type(sample)==str : sample = InputSample(full_text=sample)
        if not self.aggregation_strategy : self.aggregation_strategy = "first"
        prediction = self.transformers_model(sample.full_text, aggregation_strategy=self.aggregation_strategy)
        predictions = self.__recognizer_results_to_tags(prediction, sample)
        return predictions
    
    def batch_predict(self,dataset) -> List[List[str]]:
        predictions = []
        for data in dataset : 
            predictions.append(self.predict(data))
            if self.entity_mapping :
                  for prediction in range(len(predictions)) : 
                      for i in range(len(predictions[prediction])):
                          if predictions[prediction][i] in self.entity_mapping :
                              predictions[prediction][i] = self.entity_mapping[predictions[prediction][i]]
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
        



    

    
    

    
 


