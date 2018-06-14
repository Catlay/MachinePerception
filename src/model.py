import tensorflow as tf
import numpy as np
import utils

class BasicLSTMModel(object):
    """
    Creates training and validation computational graphs.
    Note that tf.variable_scope enables parameter sharing so that both graphs are identical.
    """
    def __init__(self, config, placeholders, mode):
        """
        Basic setup.
        :param config: configuration dictionary
        :param placeholders: dictionary of input placeholders
        :param mode: training, validation or inference
        """
        assert mode in ['training', 'validation', 'inference']
        self.config = config
        self.input_ = placeholders['input_pl']
        self.target = placeholders['target_pl']
        self.mask = placeholders['mask_pl']
        self.state_1 = placeholders['state_1']
        self.state_2 = placeholders['state_2']
        self.seq_lengths = placeholders['seq_lengths_pl']
        self.mode = mode
        self.is_training = self.mode == 'training'
        self.reuse = self.mode == 'validation'
        self.batch_size = tf.shape(self.input_)[0]  # dynamic size
        self.max_seq_length = tf.shape(self.input_)[1]  # dynamic size
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.summary_collection = 'training_summaries' if mode == 'training' else 'validation_summaries'

    def build_graph(self):
        self.build_model()
        self.build_loss()
        self.count_parameters()
        print('Count?param end')
    
    def build_model(self):
        """
        Builds the actual model.
        """
        # TODO Implement your model here
        # Some hints:
        #   1) You can access an input batch via `self.input_` and the corresponding targets via `self.target`. Note
        #      that the shape of each input and target is (batch_size, max_seq_length, input_dim)
        #
        #   2) The sequence length of each batch entry is variable, i.e. one entry in the batch might have length
        #      99 while another has length 67. No entry will be larger than what you supplied in
        #      `self.config['max_seq_length']`. This maximum sequence length is also available via `self.max_seq_length`
        #      Because TensorFlow cannot handle variable length sequences out-of-the-box, the data loader pads all
        #      batch entries with zeros so that they have size `self.max_seq_length`. The true sequence lengths are
        #      stored in `self.seq_lengths`. Furthermore, `self.mask` is a mask of shape
        #      `(batch_size, self.max_seq_length)` whose entries are 0 if this entry was padded and 1 otherwise.
        #
        #   3) You can access the config via `self.config`
        #
        #   4) The following member variables should be set after you complete this part:
        #      - `self.initial_state`: a reference to the initial state of the RNN
        #      - `self.final_state`: the final state of the RNN after the outputs have been obtained
        #      - `self.prediction`: the actual output of the model in shape `(batch_size, self.max_seq_length, output_dim)`
        
        #print('Max seq lenght',self.config['max_seq_length'])
        #print('Time',self.config['time_stamp'])
        #print('State dim',self.state_1.get_shape())
        #print('input dim',self.config['input_dim'])
        #print('output dim',self.config['output_dim'])
        
        # *****************************************************************
        # RNN with basic LSTM cell
        #******************************************************************
        with tf.variable_scope('rnn_model', reuse=self.reuse):

            cell = tf.contrib.rnn.GRUCell( self.config['hidden_states'] )
            cell = tf.contrib.rnn.MultiRNNCell( 
                [tf.contrib.rnn.GRUCell(self.config['hidden_states']) for _ in range(self.config['num_layers'] )])

            lstm = tf.contrib.rnn.BasicLSTMCell(self.config['hidden_states'])
            outputs_all = []
         
            state = self.state_1, self.state_2
            #print('Max seq lenght',self.config['max_seq_length'])
            #print('Time',self.config['time_stamp'])
            #print('State dim',self.state_1.get_shape())
            for i in range(self.config['time_stamp']):
              #print(i)
              if i>0:
               tf.get_variable_scope().reuse_variables()
              output, state = lstm(self.input_[:,i,:],state)
              print('RAW ouput',tf.convert_to_tensor(output).get_shape())
              outputs_all.append(output)
            print('shape outputs all: ', output.get_shape())
            self.final_state = state
            print ('State Final is',self.final_state.h.get_shape())
            print ('outputs all',len(outputs_all))
            outputs_all_stacked = tf.stack(outputs_all,axis=1)
            print ('outputunrst',outputs_all_stacked.get_shape())
            outputs_all_unroll = tf.reshape(outputs_all_stacked,[self.config['batch_size_dim']*(self.config['time_stamp']),self.config['hidden_states']])
            print ('outputuunroll',outputs_all_unroll.get_shape())
        
        with tf.variable_scope('output', reuse=self.reuse): 
            print('****: ', self.config['output_dim'])
            W=tf.get_variable(name='W',shape=[self.config['hidden_states'],self.config['output_dim']],initializer=tf.contrib.layers.xavier_initializer())
            b= tf.get_variable(name='b',shape=[self.config['output_dim']],initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable_scope().reuse_variables()
            print('b', b.get_shape()) # 
            print('hW',W.get_shape())
            logits=tf.matmul(outputs_all_unroll,W) + b
            self.predictions=tf.reshape(logits,[self.config['batch_size_dim'],(self.max_seq_length),self.config['output_dim']])
            print('Logits size is',logits.get_shape())
            print('finished RNN')
            predictions1= tf.argmax(logits,axis=1)
        
            print('predictions',predictions1.get_shape())
         #self.predictions=tf.reshape( predictions1,[self.config['batch_size'],(self.max_seq_length),75])
         #self.logits= logits

    def build_loss(self):
        """
        Builds the loss function.
        """
       
        # only need loss if we are not in inference mode
        if self.mode is not 'inference':
            with tf.name_scope('loss'):
                print('StartLoss')
                # TODO Implement your loss here
                # You can access the outputs of the model via `self.prediction` and the corresponding targets via
                # `self.target`. Hint 1: you will want to use the provided `self.mask` to make sure that padded values
                # do not influence the loss. Hint 2: L2 loss is probably a good starting point ...
             
                #print(' shape of mask,', self.mask.get_shape())
                mask1=tf.tile(tf.expand_dims(self.mask,axis=2), [1 , 1, self.config['output_dim']])
                #print('shape of mask: ', self.mask.get_shape())
                print('shape of prediction: ', self.predictions.get_shape())
                error= tf.abs((self.target-self.predictions) )
                print(' before change shape of error: ',error.get_shape())
                
                masked=tf.equal(mask1,0)
                zeros=tf.zeros_like(error)
                new_error=tf.where(masked,zeros,error)
                print('shape of new error: ', new_error.shape)
                                
                self.mask.get_shape()   
                self.error = error
                self.error1 = new_error
                self.masked = masked
                
                where = tf.equal(self.mask,tf.zeros_like(self.mask))
                #indices = tf.where(where)

                self.loss = tf.nn.l2_loss(new_error)
                tf.summary.scalar('loss', self.loss, collections=[self.summary_collection])
                print('build loss end')
    
    def count_parameters(self):
        """
        Counts the number of trainable parameters in this model
        """
        self.n_parameters = 0
        for v in tf.trainable_variables():
            params = 1
            for s in v.get_shape():
                params *= s.value
            self.n_parameters += params

    def get_feed_dict(self, batch):
        """
        Returns the feed dictionary required to run one training step with the model.
        :param batch: The mini batch of data to feed into the model
        :return: A feed dict that can be passed to a session.run call
        """
        input_padded, target_padded = batch.get_padded_data()
        type(input_padded)
        print( 'input padded', len(input_padded ))
        initial_state_=np.zeros([ len(input_padded ),self.config['hidden_states']],dtype=float)
       
        print('size is ',batch.seq_lengths)
        feed_dict = {self.input_: input_padded,
                     self.target: target_padded,
                     self.seq_lengths: batch.seq_lengths,
                     self.mask: batch.mask,
                     self.state_1: initial_state_,
                     self.state_2: initial_state_}
        
        return feed_dict

class MultiLSTMModel(object):
    """
    Creates training and validation computational graphs.
    Note that tf.variable_scope enables parameter sharing so that both graphs are identical.
    """
    def __init__(self, config, placeholders, mode):
        """
        Basic setup.
        :param config: configuration dictionary
        :param placeholders: dictionary of input placeholders
        :param mode: training, validation or inference
        """
        assert mode in ['training', 'validation', 'inference']
        self.config = config
        self.input_ = placeholders['input_pl']
        self.target = placeholders['target_pl']
        self.mask = placeholders['mask_pl']
        self.state_1 = placeholders['state_1']
        self.state_2 = placeholders['state_2']
        self.init_state = placeholders['init_state']
        self.seq_lengths = placeholders['seq_lengths_pl']
        self.mode = mode
        self.is_training = self.mode == 'training'
        self.reuse = self.mode == 'validation'
        self.batch_size = tf.shape(self.input_)[0]  # dynamic size
        self.max_seq_length = tf.shape(self.input_)[1]  # dynamic size
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.summary_collection = 'training_summaries' if mode == 'training' else 'validation_summaries'

    def build_graph(self):
        self.build_model()
        self.build_loss()
        self.count_parameters()
        print('Count?param end')

    def build_cell(self, lstm_size, keep_prob):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(cell=lstm, output_keep_prob=keep_prob)
        return drop   

    def make_cell(self, lstm_size):
        return tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)    
    
    def build_model(self):
        """
        Builds the actual model.
        """
        #******************************************************************
        # RNN: Multilayer LSTM
        #******************************************************************
        with tf.variable_scope('rnn_model', reuse=self.reuse):         

            print('Time',self.config['time_stamp'])

            state_per_layer_list = tf.unstack(self.init_state, axis=0)
            rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], 
                    state_per_layer_list[idx][1]) for idx in range(self.config['num_layers'])])
            #print('rnn tuple state dim: ', rnn_tuple_state)           

            mcell = tf.nn.rnn_cell.MultiRNNCell(
                [self.build_cell(
                    self.config['hidden_states'],
                    self.config['keep_prob']) for _ in range(self.config['num_layers'])],
                    state_is_tuple=True)
            
            output, state = tf.nn.dynamic_rnn(mcell, self.input_, dtype=tf.float32, initial_state=rnn_tuple_state)
            print('shape output all: ', output.get_shape())

            states_all_stacked = tf.stack(state,axis=0)
            print ('states stacked',states_all_stacked.get_shape())
            self.final_state = states_all_stacked
            print ('shape state Final is',self.final_state)
  
            outputs_all_unroll = tf.reshape(output,[self.config['batch_size_dim']*(self.config['time_stamp']),self.config['hidden_states']])
            print ('outputuunroll',outputs_all_unroll.get_shape())
         

        with tf.variable_scope('output', reuse=self.reuse): 
            W = tf.get_variable(name='W',shape=[self.config['hidden_states'],self.config['output_dim']],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b',shape=[self.config['output_dim']],initializer=tf.contrib.layers.xavier_initializer())
            print('b', b.get_shape())  
            print('hW',W.get_shape())
            tf.get_variable_scope().reuse_variables()
            logits = [tf.matmul(outputs_all_unroll, W) + b]
            #print('Logits size is',logits)
            
            self.predictions=tf.reshape(logits,[self.config['batch_size_dim'],(self.max_seq_length),self.config['output_dim']])
            
            print('predictions',self.predictions.get_shape())          
            print('finished RNN')

    def build_loss(self):
        """
        Builds the loss function.
        """
       
        # only need loss if we are not in inference mode
        if self.mode is not 'inference':
            with tf.name_scope('loss'):
                print('StartLoss')

                # TODO: if we have one-hot encoding, we need to get rid of those columns to calculate error
                #print('prediction: ', self.predictions)
                #print('target: ', self.target)
                print('shape of prediction: ', self.predictions.get_shape())
                mask1=tf.tile(tf.expand_dims(self.mask,axis=2), [1 , 1, self.config['output_dim']])             
                error= tf.abs((self.target-self.predictions) )            
                masked=tf.equal(mask1,0)
                zeros=tf.zeros_like(error)
                new_error=tf.where(masked,zeros,error)
                print('shape of error: ', new_error.get_shape())              
                
                self.mask.get_shape()   
                self.error = error
                self.error1 = new_error
                self.masked = masked
                
                where = tf.equal(self.mask,tf.zeros_like(self.mask))

                self.loss = tf.nn.l2_loss(new_error)
                tf.summary.scalar('loss', self.loss, collections=[self.summary_collection])
                print('build loss end')
    
    def count_parameters(self):
        """
        Counts the number of trainable parameters in this model
        """
        self.n_parameters = 0
        for v in tf.trainable_variables():
            params = 1
            for s in v.get_shape():
                params *= s.value
            self.n_parameters += params

    def get_feed_dict(self, batch):
        """
        Returns the feed dictionary required to run one training step with the model.
        :param batch: The mini batch of data to feed into the model
        :return: A feed dict that can be passed to a session.run call
        """
        input_padded, target_padded = batch.get_padded_data()
        type(input_padded)
        print( 'input padded', len(input_padded ))
        initial_state_= np.zeros([ len(input_padded ),self.config['hidden_states']],dtype=float)
        initial_states_= np.zeros((self.config['num_layers'], 2, self.config['batch_size'], self.config['hidden_states']),dtype=float)
    
       
        print('size is ',batch.seq_lengths)
        feed_dict = {self.input_: input_padded,
                     self.target: target_padded,
                     self.seq_lengths: batch.seq_lengths,
                     self.mask: batch.mask,
                     self.state_1: initial_state_,
                     #self.state_2: initial_state_,
                     self.init_state: initial_states_}
        
        return feed_dict

class Seq2SeqModel(object):
    """
    Creates training and validation computational graphs.
    Note that tf.variable_scope enables parameter sharing so that both graphs are identical.
    """
    def __init__(self, config, placeholders, mode):
        """
        Basic setup.
        :param config: configuration dictionary
        :param placeholders: dictionary of input placeholders
        :param mode: training, validation or inference
        """
        assert mode in ['training', 'validation', 'inference']
        self.config = config
        self.input_ = placeholders['input_pl']
        self.target = placeholders['target_pl']
        self.mask = placeholders['mask_pl']
        self.state_1 = placeholders['state_1']
        self.state_2 = placeholders['state_2']
        self.seq_lengths = placeholders['seq_lengths_pl']
        self.mode = mode
        self.is_training = self.mode == 'training'
        self.reuse = self.mode == 'validation'
        self.batch_size = tf.shape(self.input_)[0]  # dynamic size
        self.max_seq_length = tf.shape(self.input_)[1]  # dynamic size
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.summary_collection = 'training_summaries' if mode == 'training' else 'validation_summaries'

        self.enc_in = enc_in
        self.dec_in = dec_in
        self.dec_out = dec_out        

    def build_graph(self):
        self.build_model()
        self.build_loss()
        self.count_parameters()
        print('Count?param end')
    
    def make_cell(self, lstm_size):
        return tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)    

    def build_cell(self, lstm_size, keep_prob):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(cell=lstm, output_keep_prob=keep_prob)
        return drop 

    def build_model(self):
        """
        Builds the actual model.
        """
        #******************************************************************
        # RNN: Seq2Seq
        #******************************************************************
        
        # === Transform the inputs ===
        with tf.name_scope("inputs"):

            enc_in = tf.transpose(enc_in, [1, 0, 2])
            dec_in = tf.transpose(dec_in, [1, 0, 2])
            dec_out = tf.transpose(dec_out, [1, 0, 2])

            enc_in = tf.reshape(enc_in, [-1, self.input_size])
            dec_in = tf.reshape(dec_in, [-1, self.input_size])
            dec_out = tf.reshape(dec_out, [-1, self.input_size])

            enc_in = tf.split(enc_in, source_seq_len-1, axis=0)
            dec_in = tf.split(dec_in, target_seq_len, axis=0)
            dec_out = tf.split(dec_out, target_seq_len, axis=0)

        with tf.variable_scope('rnn_model', reuse=self.reuse):
            
            print('Time',self.config['time_stamp'])

            state_per_layer_list = tf.unstack(self.init_state, axis=0)
            rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], 
                    state_per_layer_list[idx][1]) for idx in range(self.config['num_layers'])])
            #print('rnn tuple state dim: ', rnn_tuple_state)           

            encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.build_cell(
                    self.config['hidden_states'],
                    self.config['keep_prob']) for _ in range(self.config['num_layers'])],
                    state_is_tuple=True)
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.build_cell(
                    self.config['hidden_states'],
                    self.config['keep_prob']) for _ in range(self.config['num_layers'])],
                    state_is_tuple=True)        
            """
            encoder_cell = tf.contrib.rnn.BasicLSTMCell(self.config['hidden_states']) # Build enocder cell
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config['hidden_states']) # Build decoder cell
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths, time_major=True) # Helper
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer) # Decoder
            outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...) # Dynamic decoding
            logits = outputs.rnn_output
            """

            # Run Dynamic RNN
            #   encoder_outputs: [max_time, batch_size, num_units]
            #   encoder_state: [batch_size, num_units]
            _, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, self.input_, dtype=tf.float32, initial_state=rnn_tuple_state)

            # Decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, encoder_state)
            # Dynamic decoding
            outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
            logits = outputs.rnn_output

            _, enc_state = tf.contrib.rnn.static_rnn(cell, enc_in, dtype=tf.float32) # Encoder
            output, self.states = tf.contrib.legacy_seq2seq.rnn_decoder( dec_in, enc_state, mcell, loop_function=lf ) # Decoder
            
            # **************************MultiCell LSTM**************** 
            states_all_stacked = tf.stack(state,axis=1)
            print ('states stacked',states_all_stacked.get_shape())
            self.final_state = states_all_stacked
            print ('shape state Final is',self.final_state)
  
            outputs_all_unroll = tf.reshape(output,[self.config['batch_size_dim']*(self.config['time_stamp']),self.config['hidden_states']])
            print ('outputuunroll',outputs_all_unroll.get_shape())
            # ******************************************  

        with tf.variable_scope('output', reuse=self.reuse): 
            W = tf.get_variable(name='W',shape=[self.config['hidden_states'],self.config['output_dim']],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b',shape=[self.config['output_dim']],initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable_scope().reuse_variables()
            logits=tf.matmul(outputs_all_unroll,W) + b
            self.predictions=tf.reshape(logits,[self.config['batch_size_dim'],(self.max_seq_length),self.config['output_dim']])
            print('finished RNN')

    def build_loss(self):
        """
        Builds the loss function.
        """
       
        # only need loss if we are not in inference mode
        if self.mode is not 'inference':
            with tf.name_scope('loss'):
                print('StartLoss')

                # TODO: if we have one-hot encoding, we need to get rid of those columns to calculate error
                #print('prediction: ', self.predictions)
                #print('target: ', self.target)
                print('shape of prediction: ', self.predictions.get_shape())
                mask1=tf.tile(tf.expand_dims(self.mask,axis=2), [1 , 1, self.config['output_dim']])             
                error= tf.abs((self.target-self.predictions) )            
                masked=tf.equal(mask1,0)
                zeros=tf.zeros_like(error)
                new_error=tf.where(masked,zeros,error)
                print('shape of error: ', new_error.get_shape())              
                
                self.mask.get_shape()   
                self.error = error
                self.error1 = new_error
                self.masked = masked
                
                where = tf.equal(self.mask,tf.zeros_like(self.mask))

                self.loss = tf.nn.l2_loss(new_error)
                tf.summary.scalar('loss', self.loss, collections=[self.summary_collection])
                print('build loss end')
    
    def count_parameters(self):
        """
        Counts the number of trainable parameters in this model
        """
        self.n_parameters = 0
        for v in tf.trainable_variables():
            params = 1
            for s in v.get_shape():
                params *= s.value
            self.n_parameters += params

    def get_feed_dict(self, batch):
        """
        Returns the feed dictionary required to run one training step with the model.
        :param batch: The mini batch of data to feed into the model
        :return: A feed dict that can be passed to a session.run call
        """
        input_padded, target_padded = batch.get_padded_data()
        type(input_padded)
        print( 'input padded', len(input_padded ))
        initial_state_= np.zeros([ len(input_padded ),self.config['hidden_states']],dtype=float)
        initial_states_= np.zeros((self.config['num_layers'], 2, self.config['batch_size'], self.config['hidden_states']),dtype=float)
    
       
        print('size is ',batch.seq_lengths)
        feed_dict = {self.input_: input_padded,
                     self.target: target_padded,
                     self.seq_lengths: batch.seq_lengths,
                     self.mask: batch.mask,
                     self.state_1: initial_state_,
                     #self.state_2: initial_state_,
                     self.init_state: initial_states_}
        
        return feed_dict