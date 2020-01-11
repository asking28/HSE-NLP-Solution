import os
import tensorflow as tf
import keras
import os
from utils import *
import numpy as np
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import string

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

class DialogueManager(object):
    def __init__(self, paths):
		
        print("Loading resources...")
        self.tokenizer = unpickle_file(paths['TOKENIZER'])
        self.embedding_dim =200
        self.units = 1024
        self.vocab_size = len(self.tokenizer.word_index)+1
        self.BATCH_SIZE=256
        self.encoder = Encoder(self.vocab_size, self.embedding_dim, self.units, self.BATCH_SIZE)
        self.decoder = Decoder(self.vocab_size, self.embedding_dim, self.units, self.BATCH_SIZE)
            #self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you, thank you: https://stackoverflow.com/questions/%s'
        self.optimizer = tf.keras.optimizers.Adam()
        self.checkpoint_dir = "checkpoint"
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder,
                                               decoder=self.decoder)
        # Goal-oriented part:
        #self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        #self.thread_ranker = ThreadRanker(paths)
        
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        #self.create_chitchat_bot()
    def evaluate(self,sentence):
		
	  #attention_plot = np.zeros((max_len,max_len))
      print(tf.train.latest_checkpoint(self.checkpoint_dir))
      max_len=13
      sentence=sentence.lower()
      sentence=sentence.translate(str.maketrans('', '', string.punctuation))
      sentence=sentence.rstrip().strip()
	    
      sentence = "<start> "+sentence+" <end>"

      inputs = [self.tokenizer.word_index[i] for i in sentence.split(' ') if i in self.tokenizer.word_index]
      inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                maxlen=max_len,padding="post",truncating="post")
      inputs = tf.convert_to_tensor(inputs)

      result = ''

      hidden = [tf.zeros((1, self.units))]
      enc_out, enc_hidden = self.encoder(inputs, hidden)

      dec_hidden = enc_hidden
      dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)

      for t in range(max_len):

          predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                      dec_hidden,
                                      enc_out)

        # storing the attention weights to plot later on
        #attention_weights = tf.reshape(attention_weights, (-1, ))
        #attention_plot[t] = attention_weights.numpy()

          predicted_id = tf.argmax(predictions[0]).numpy()
          if self.tokenizer.index_word[predicted_id] == '<end>':
            return result
          result += self.tokenizer.index_word[predicted_id] + ' '
        #print(result)


        # the predicted ID is fed back into the model
          dec_input = tf.expand_dims([predicted_id], 0)

      return result
        
       
    def generate_answer(self, sentence):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        print(sentence)
        result = self.evaluate(sentence)
        return result
