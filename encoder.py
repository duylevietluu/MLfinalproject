import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout, LayerNormalization, Dense, MultiHeadAttention, Flatten, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# tokenize and pad sequences
def tokenize_and_pad(texts, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded

# Positional Encoding Function
def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / float(d_model))
        return pos * angles
    
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# Encoder Layer Class
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dropout1 = Dropout(rate)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.dropout2 = Dropout(rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        
    def call(self, x, training):
        # self-attention layer & dropout 1 & add and normalize 1
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)
        
        # feedforward layer & dropout 2 & add and normalize 2
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)

        return out2

# Encoder Class
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_length, rate=0.1):
        """
        num_layers: number of encoder layers
        d_model: dimension of the model
        num_heads: number of attention heads
        dff: dimension of feedforward network
        input_vocab_size: size of input vocabulary
        max_length: maximum position encoding
        rate: dropout rate
        """
        super(Encoder, self).__init__()

        # initialize parameters
        self.d_model = d_model
        self.num_layers = num_layers
        
        # init embedding and positional encoding layers
        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_length, self.d_model)
        
        # init encoder layers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        # init dropout layer
        self.dropout = Dropout(rate)
        
    def call(self, x, training):
        # x: (batch_size, seq_len)
        seq_len = tf.shape(x)[1]

        # convert token indices to embeddings of dimension d_model; scale by sqrt(d_model) to normalize variance
        x = self.embedding(x) # (batch_size, seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # add positional encoding to embeddings
        x += self.pos_encoding[:, :seq_len, :]

        # apply dropout to embeddings to prevent overfitting
        x = self.dropout(x, training=training)
        
        # pass embeddings through each encoder layer in turn
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training)

        # return x, (batch_size, seq_len, d_model)
        return x

class EncoderBooleanTransformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate=0.1):
        super(EncoderBooleanTransformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.global_pool = GlobalAveragePooling1D()
        self.intermediate_layer = Dense(64, activation='relu') 
        self.final_layer = Dense(1, activation='sigmoid')
        
    def call(self, x, training):
        enc_output = self.encoder(x, training)  # 3D (batch_size, inp_seq_len, d_model)
        pooled_output = self.global_pool(enc_output)  # 2D (batch_size, d_model)
        intermediate_output = self.intermediate_layer(pooled_output) # (batch_size, 64)
        final_output = self.final_layer(intermediate_output)  # (batch_size, 1)
        return final_output
    
class EncoderCategoryTransformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, num_classes, rate=0.1):
        super(EncoderCategoryTransformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.global_pool = GlobalAveragePooling1D()
        self.intermediate_layer = Dense(64, activation='relu') 
        self.final_layer = Dense(num_classes, activation='softmax')

    def call(self, x, training):
        enc_output = self.encoder(x, training) # 3D (batch_size, inp_seq_len, d_model)
        pooled_output = self.global_pool(enc_output) # 2D (batch_size, d_model)
        intermediate_output = self.intermediate_layer(pooled_output) # (batch_size, 64)
        final_output = self.final_layer(intermediate_output) # (batch_size, num_classes)
        return final_output

# test run on imdb dataset
def imdb_test():
    # Hyperparameters
    num_layers = 2
    d_model = 256
    num_heads = 8
    dff = 512
    num_words = 10000
    input_vocab_size = num_words + 1
    max_length = 120

    # dataset
    import datasets
    imdb = datasets.load_dataset('imdb')

    # tokenizer
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(imdb['train']['text'])

    # data split
    X_train = tokenize_and_pad(imdb['train']['text'], tokenizer, max_length)
    y_train = np.array(imdb['train']['label'])

    X_test = tokenize_and_pad(imdb['test']['text'], tokenizer, max_length)
    y_test = np.array(imdb['test']['label'])

    # model
    model = EncoderBooleanTransformer(num_layers, d_model, num_heads, dff, input_vocab_size, max_length)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # train
    model.fit(X_train, y_train, batch_size=32, epochs=2, validation_data=(X_test, y_test))

    # evaluate
    model.evaluate(X_test, y_test)

# test run on goemotion dataset
def goemotion_test():
    # Hyperparameters
    num_layers = 2
    d_model = 256
    num_heads = 8
    dff = 512
    num_words = 10000
    input_vocab_size = num_words + 1
    max_length = 120
    num_classes = 28

    # dataset, filter out samples with multiple labels
    import datasets
    goemotion = datasets.load_dataset('go_emotions')
    goemotion = goemotion.filter(lambda x: len(x["labels"]) == 1)

    # tokenizer
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(goemotion['train']['text'])

    # transform labels
    from sklearn.preprocessing import MultiLabelBinarizer
    def one_hot_encode(labels, num_classes):
        mlb = MultiLabelBinarizer()
        mlb.fit([list(range(num_classes))])
        return mlb.transform(labels)

    # data split
    X_train = tokenize_and_pad(goemotion['train']['text'], tokenizer, max_length)
    y_train = one_hot_encode(goemotion['train']['labels'], num_classes)

    X_val = tokenize_and_pad(goemotion['validation']['text'], tokenizer, max_length)
    y_val = one_hot_encode(goemotion['validation']['labels'], num_classes)

    X_test = tokenize_and_pad(goemotion['test']['text'], tokenizer, max_length)
    y_test = one_hot_encode(goemotion['test']['labels'], num_classes)

    # model
    model = EncoderCategoryTransformer(num_layers, d_model, num_heads, dff, input_vocab_size, max_length, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train
    model.fit(X_train, y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val))

    # evaluate
    model.evaluate(X_test, y_test)


if __name__ == '__main__':
    goemotion_test()