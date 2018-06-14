import os
import tensorflow as tf
import numpy as np
import utils

from config import test_config
from visualize import visualize_joint_angles
from utils import export_to_csv
from train import load_data, get_model_and_placeholders


def main(config):
    # load the data
    data_test = load_data(config, 'test')
    """
    if config['preprocess'] :
        data_train = load_data(config, 'train')
        data_mean, data_std, dim_to_ignore, dim_to_use = utils.standardization_stats(data_train.input_)

        test_set = []
        # RAH Standardization -- subtract mean, divide by stdev
        test_set  = utils.standardize_data( data_test.input_, data_mean, data_std, dim_to_ignore, config['one_hot'] )
        data_test.input_ = test_set
        #print('shape data_test.input_: ', test_set)
        print("standardization of test data done.")
    """
    
    print('Size of input',data_test.input_[0].shape)
    config['input_dim'] = config['output_dim'] = data_test.input_[0].shape[-1]
    rnn_model, placeholders = get_model_and_placeholders(config)

    # restore the model by first creating the computational graph
    with tf.name_scope('inference'):
        rnn_model = rnn_model(config, placeholders, mode='inference')
        rnn_model.build_graph()

    with tf.Session() as sess:
        # now restore the trained variables
        # this operation will fail if this `config` does not match the config you used during training
        saver = tf.train.Saver()
        ckpt_id = config['checkpoint_id']
        if ckpt_id is None:
            ckpt_path = tf.train.latest_checkpoint(config['model_dir'])
        else:
            ckpt_path = os.path.join(os.path.abspath(config['model_dir']), 'model-{}'.format(ckpt_id))
        print('Checkpointid ',ckpt_path)
        print('Evaluating ' + ckpt_path)
        saver.restore(sess, ckpt_path)

        # loop through all the test samples
        seeds = []
        predictions = []
        ids = []
        print(config['batch_size'])
        print(config['hidden_states'])
        initial_state = np.zeros([1, config['hidden_states']])
        initial_states = np.zeros((config['num_layers'], 2, 1, config['hidden_states']))
          
        j=0
        
        for batch in data_test.all_batches():
            
         ids.extend(batch.ids)
         print('Extended with',len(ids))
         j=j+1
         input_all = np.array(batch.input_)
         seeds.append(input_all)
         
         seq_leng = np.array(batch.seq_lengths)
         print('seq len all',seq_leng.shape)
         for i in range(batch.batch_size):
            # initialize the RNN with the known sequence (here 2 seconds)
            # no need to pad the batch because in the test set all batches have the same length
            input_=input_all[i]

            #print('Input shape',input_.shape)
           
            #print('shape seq',seq_leng[i].size)
            #print('type',len([input_[0]]))
            # here we are requesting the final state as we later want to supply this back into the RNN
            # this is why the model should have a member `self.final_state`
            fetch = [rnn_model.final_state,rnn_model.predictions]
            feed_dict = {placeholders['input_pl']: [[input_[0]]],
                         placeholders['seq_lengths_pl']: [1],
                         placeholders['state_1']:initial_state, 
                         placeholders['state_2']:initial_state,
                         placeholders['init_state']:initial_states}

            [states,predictions_] = sess.run(fetch, feed_dict)
            print(input_[1:][:].shape)
            for frame in range(len(input_[1:][:])):
             state_array = np.array(states) 
             fetch = [rnn_model.final_state,rnn_model.predictions]
             #print('fetch: ', fetch)
             feed_dict = {placeholders['input_pl']: [[input_[frame]]],
                         placeholders['seq_lengths_pl']: [1],
                         #placeholders['state_1']:state_array[0], 
                         #placeholders['state_2']:state_array[1],
                         placeholders['init_state']:state_array}

             [states,predictions_] = sess.run(fetch, feed_dict)
            # now get the prediction by predicting one pose at a time and feeding this pose back into the model to
            # get the prediction for the subsequent time step
            next_pose = predictions_
            
            predicted_poses = []
            for f in range(config['prediction_length']):
                # TODO evaluate your model here frame-by-frame
                # To do so you should
                #   1) feed the previous final state of the model as the next initial state
                #   2) feed the previous output pose of the model as the new input (single frame only)
                #   3) fetch both the final state and prediction of the RNN model that are then re-used in the next
                #      iteration
                state_array = np.array(states)
                #print('Shape of state array',state_array.shape)
                #print('integer trying',next_pose.shape[1])
                fetch = [rnn_model.final_state,rnn_model.predictions]

               
                feed_dict = {placeholders['input_pl']:next_pose,
                            placeholders['seq_lengths_pl']: [1],
                            #placeholders['state_1']:state_array[0],
                            #placeholders['state_2']:state_array[1],
                            placeholders['init_state']:state_array}

                #print('State dim0', state_array[0].shape)
                #print('State dim0', state_array[1].shape)
                #print('State is', state_array)
                #print('next  pose is', next_pose)
                [states, predicted_pose] = sess.run(fetch, feed_dict)
                #print('pred pose is', predicted_pose)
                predicted_poses.append(np.copy(predicted_pose))
                next_pose = predicted_pose
                    
            predicted_poses = np.concatenate(predicted_poses, axis=1)
            print('predicted poses shape', predicted_poses.shape)
            
            if config['preprocess'] :
                predicted_poses = utils.unStandardizeData(
                    data_train, predicted_poses, data_mean, data_std, dim_to_ignore, config['one_hot'])
                predicted_poses = np.reshape(predicted_poses, (1,25,75))    
                predictions.append(predicted_poses)  
            else:
                predictions.append(predicted_poses) 
           
            
   
        #print('Bstchids',len(ids))
        print('J is',j)
        print('All predictions are',len(predictions))
        seeds = np.concatenate(seeds, axis=0)
        predictions = np.concatenate(predictions, axis=0)
    print('Extended with',len(ids))
    print('seeds shape is: ', seeds.shape)
    print('All predictions are',predictions.shape)

    # the predictions are now stored in test_predictions, you can do with them what you want
    # for example, visualize a random entry
    idx = np.random.randint(0, len(seeds))
    seed_and_prediction = np.concatenate([seeds[idx], predictions[idx]], axis=0)
    visualize_joint_angles([seed_and_prediction], change_color_after_frame=seeds[0].shape[0])

    # or, write out the test results to a csv file that you can upload to Kaggle
    model_name = config['model_dir'].split('/')[-1]
    model_name = config['model_dir'].split('/')[-2] if model_name == '' else model_name
    output_file = os.path.join(config['model_dir'], 'submit_to_kaggle_{}_{}.csv'.format(config['prediction_length'], model_name))
    print(predictions.shape)
    print(len(ids))
    export_to_csv(predictions, ids, output_file)


if __name__ == '__main__':
    main(test_config)
