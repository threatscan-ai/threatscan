from transformers import AutoImageProcessor, VideoMAEImageProcessor, VideoMAEForVideoClassification

# other models => "MCG-NJU/videomae-base" 
def load_model(label2id, id2label, model_name="MCG-NJU/videomae-base-finetuned-kinetics"):
    processor = AutoImageProcessor.from_pretrained(model_name)
    print(label2id)
    model = VideoMAEForVideoClassification.from_pretrained(model_name, 
                                                           label2id=label2id,
                                                            id2label=id2label,
                                                            ignore_mismatched_sizes=True)
    return model, processor