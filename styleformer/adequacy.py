class Adequacy():

  def __init__(self, model_tag='prithivida/parrot_adequacy_on_BART'):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_tag)
    self.tokenizer = AutoTokenizer.from_pretrained(model_tag)
    
  def filter(self, input_phrase, para_phrases, adequacy_threshold, device="cpu"):
      top_adequacy_phrases = []
      for para_phrase in para_phrases:
        x = self.tokenizer.encode(input_phrase, para_phrase, return_tensors='pt',truncation_strategy='only_first')
        self.nli_model = self.nli_model.to(device)
        logits = self.nli_model(x.to(device))[0]
        # we throw away "neutral" (dim 1) and take the probability of "entailment" (2) as the adequacy score
        entail_contradiction_logits = logits[:,[0,2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:,1]
        adequacy_score = prob_label_is_true[0].item()
        if adequacy_score >= adequacy_threshold:
            top_adequacy_phrases.append(para_phrase)
      return top_adequacy_phrases

  def score(self, input_phrase, para_phrases, adequacy_threshold, device="cpu"):
      adequacy_scores = {}
      for para_phrase in para_phrases:
        x = self.tokenizer.encode(input_phrase, para_phrase, return_tensors='pt',truncation_strategy='only_first')
        self.nli_model = self.nli_model.to(device)
        logits = self.nli_model(x.to(device))[0]
        # we throw away "neutral" (dim 1) and take the probability of "entailment" (2) as the adequacy score
        entail_contradiction_logits = logits[:,[0,2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:,1]
        adequacy_score = prob_label_is_true[0].item()
        if adequacy_score >= adequacy_threshold:
          adequacy_scores[para_phrase] = adequacy_score
      return adequacy_scores      