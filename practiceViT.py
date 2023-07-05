from huggingface_hub import notebook_login
from datasets import load_dataset

from transformers import(
    AutoImageProcessor,
    DefaultDataCollator, # does not apply preprocessing
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer,
    create_optimizer,
    TFAutoModelForImageClassification,
    pipeline,
)

import torch
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

#for augmentations
import cv2
import albumentations

import numpy as np
import tensorflow as tf
from PIL import Image

import evaluate

accuracy = evaluate.load("accuracy")

token='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'
notebook_login()

device = 'cuda'
model_name = "DatasetModel"
dataset_name = "Kaludi/data-food-classification"

'''
EVALUATION
''' 
#takes predicted labels and true labels
#converts predicted probabiliteds to class labels
#computes accuracy metric using predicted and true labels
def compute_metrics(eval_pred): #Tuple 
    #unpack into prediction and label
    predictions, labels = eval_pred
    #Find the index of the highest prediction value for each example
    predictions = np.argmax(predictions, axis=1)
    #compute accuracy
    return accuracy.compute(predictions=predictions, references=labels)



#create helper function to apply transforms 
def transforms(examples):
    # turns img into rgb format and does _transforms
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]

    # removes the current example from the examples dictionary
    # so we cant use this img for another example
    del examples["image"]

    # Returns the modified examples dictionary
    return examples

#read pipeline score
def get_pred(score_labelList): #input classifier
    maxScore = score_labelList[0]['score']
    highestLabel = score_labelList[0]['label']
    for i, index in enumerate(score_labelList):
        if score_labelList[i]['score'] < maxScore:
            maxScore = score_labelList[i]['score']
            highestLabel = score_labelList[i]['label']
    #return highest label 
    return highestLabel

def findAcc():
    numclasses = len(labels)
    # track test loss
    class_correct = list(0. for i in range(numclasses))
    class_total = list(0. for i in range(numclasses))


    # iterate over test data
    for i, img in enumerate(ds["image"]):
        # convert output probabilities to predicted class
        pred_label = get_pred(classifier(img))

        for i in range(numclasses):
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                ds["label"][i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))

    
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))




#load dataset (first 5000 images)
dataset = load_dataset(dataset_name, split="train[:5000]")

#20% in test / 80% in train
dataset = dataset.train_test_split(test_size=0.2)

#create dictionary for labels
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    #label2id and id2label are inverse of eachother
    label2id[label] = str(i)
    id2label[str(i)] = label

#load ViT img processor to turn img to tensor
#load pretrained checkpoint
checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

#normalize img
normalize = Normalize(mean=image_processor.image_mean, 
                      std=image_processor.image_std)

size = (
    #turns img into square
    image_processor.size["shortest_edge"]
    #checks if "shortest_edge" exists
    if "shortest_edge" in image_processor.size
    #otherwise the img size is determined by h and w
    else (image_processor.size["height"], image_processor.size["width"])
)

#crops random part of img, rezises, and normalizes with image mean and std deviation
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])



#apply transform on whole dataset
dataset = dataset.with_transform(transforms)

# #create batch of examples using DefaultDataCollator
data_collator = DefaultDataCollator()



'''
Training 
'''
#auto model from pretrained checkpoint
model = AutoModelForImageClassification.from_pretrained(
    checkpoint, #using google/vit-base-patch16-224-in21k
    num_labels=len(labels), # get num labels
    id2label=id2label, #dictionary that maps class id to label
    label2id=label2id,
)


#hyper Parameters
#pass into trainier
training_args = TrainingArguments(
    output_dir=model_name,
    remove_unused_columns=False, #need this to create pixel_values
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    #pass in datasets
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()
#takes too long



'''
inference
'''

ds = load_dataset(dataset_name, split="validation[:10]")
image = ds["image"][0]
print(image)
print(ds)

#pass in model into pipeline()
classifier = pipeline("image-classification", model=model_name)

#load image processor to preprocess image and return input as tensor
# use image processor model
image_processor = AutoImageProcessor.from_pretrained(model_name)
#preprocess images through image_processor
inputs = image_processor(image, return_tensors="pt")

# Pass your inputs to the model and return the logits:
model = AutoModelForImageClassification.from_pretrained(model_name)
with torch.no_grad():
    logits = model(**inputs).logits

print(logits)
trainer.push_to_hub()