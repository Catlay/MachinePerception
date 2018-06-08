import tensorflow as tf
import numpy as np


class RNNModel(object):
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
        #print('Count?param end')
    
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
        print('Max seq lenght',self.config['max_seq_length'])
        print('Time',self.config['time_stamp'])
        print('State dim',self.state_1.get_shape())
        
        with tf.variable_scope('rnn_model', reuse=self.reuse):
            #self.initial_state = None
            #self.final_state = None
            #self.prediction = None
            lstm=tf.contrib.rnn.BasicLSTMCell(self.config['hidden_states'])
            outputs_all=[]
         
            state=self.state_1, self.state_2
            print('Max seq lenght',self.config['max_seq_length'])
            print('Time',self.config['time_stamp'])
            print('State dim',self.state_1.get_shape())
            for i in range(self.config['time_stamp']):
              print(i)
              if i>0:
               tf.get_variable_scope().reuse_variables()
              output,state=lstm(self.input_[:,i,:],state)
              print('RAW ouput',tf.convert_to_tensor(output).get_shape())
              outputs_all.append(output)
            self.final_state=state
              #print ('State Final is',self.final_state.h.get_shape())
            print ('outputs all',len(outputs_all))
            outputs_all_stacked=tf.stack(outputs_all,axis=1)
            print ('outputunrst',outputs_all_stacked.get_shape())
            outputs_all_unroll=tf.reshape(outputs_all_stacked,[self.config['batch_size_dim']*(self.config['time_stamp']),self.config['hidden_states']])
            print ('outputuunroll',outputs_all_unroll.get_shape())
        with tf.variable_scope('output', reuse=self.reuse): 
            W=tf.get_variable(name='W',shape=[self.config['hidden_states'],self.config['output_dim']],initializer=tf.contrib.layers.xavier_initializer())
            b= tf.get_variable(name='b',shape=[self.config['output_dim']],initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable_scope().reuse_variables()
            print(b.get_shape())
            print('hW',W.get_shape()) 
            logits=tf.matmul(outputs_all_unroll,W) + b
            self.predictions=tf.reshape(logits,[self.config['batch_size_dim'],(self.max_seq_length),self.config['output_dim']])
            print('Logits size is',logits.get_shape())
            print('finished RNN')
            predictions1= tf.argmax(logits,axis=1)
        
            print('predictions',predictions1.get_shape())
         #self.predictions=tf.reshape( predictions1,[self.config['batch_size'],(self.max_seq_length),75])


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
              #  loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits)
              #  self.loss = tf.reduce_sum(loss)
                print(' shape of mask,', self.mask.get_shape())
                mask1=tf.tile(tf.expand_dims(self.mask,axis=2), [1 , 1, self.config['output_dim']])
                print(mask1.get_shape())
                error= tf.abs((self.target-self.predictions) )
                print('er before change',error.get_shape())
                
                masked=tf.greater(mask1,0)
                zeros=tf.zeros_like(error)
                new_error=tf.where(masked,zeros,error)
                
                
                self.mask.get_shape()
        
                print('error after ', new_error.get_shape())
                self.loss=tf.nn.l2_loss(new_error)
                print('loss ', self.loss.get_shape())
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
       
        
        feed_dict = {self.input_: input_padded,
                     self.target: target_padded,
                     self.seq_lengths: batch.seq_lengths,
                     self.mask: batch.mask,
                     self.state_1: initial_state_,
                     self.state_2: initial_state_}
        
        return feed_dict
