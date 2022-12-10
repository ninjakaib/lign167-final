from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import torch
import flair

#utilize M1 metal performance shaders for training
flair.device = torch.device("mps")

# this is the folder in which train, test and dev files reside
data_folder = './newsdata/text'

label_type = 'sentiment'

# load corpus containing training, test and dev data
corpus: Corpus = ClassificationCorpus(data_folder,
                                      test_file='test.txt',
                                      dev_file='dev.txt',
                                      train_file='train.txt',
                                      label_type = label_type)

# create the label dictionary
label_dict = corpus.make_label_dictionary(label_type=label_type)

# initialize transformer document embeddings (many models are available)
document_embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune=True)

# create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type=label_type)

# initialize trainer
trainer = ModelTrainer(classifier, corpus)

# run training with fine-tuning
path = 'resources/taggers/question-classification-with-transformer'
trainer.fine_tune(path,
                  learning_rate=5.0e-5,
                  mini_batch_size=4,
                  max_epochs=10,
                  checkpoint=True
                  )

# continue training later
trained_model = TextClassifier.load(path+'/checkpoint.pt')

trainer.resume(trained_model,
               base_path=path+'-resume',
               max_epochs=25
               )

