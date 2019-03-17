'''
Train individual autoencoders, for 8 classes of shapenet objects.

'''

# make sure we can import
import sys
sys.path.append("/home/ubuntu/")

import os
import os.path as osp

from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.general_utils import plot_3d_point_cloud
import numpy as np

top_out_dir = '../data/'          # Use to save Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

n_pc_points = 2048                # Number of points per model.
bneck_size = 128                  # Bottleneck-AE size
ae_loss = 'chamfer'               # Loss to optimize: 'emd' or 'chamfer'
class_names = ["airplane",
               "cabinet",
               "car",
               "chair",
               "lamp",
               "sofa",
               "table",
               "vessel"]

# experiment name
experiment_name = 'multi_class'

for class_name in class_names:

    # load point cloud data for that class
    syn_id = snc_category_to_synth_id()[class_name]
    class_dir = osp.join(top_in_dir , syn_id)
    if class_name=='airplane':
        all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)
    else:
        pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)
        assert all_pc_data.n_points == pc_data.n_points
        all_pc_data.num_examples += pc_data.num_examples
        #all_pc_data.labels = np.vstack((all_pc_data.labels, pc_data.labels))
        all_pc_data.labels.append(pc_data.labels)
        all_pc_data.point_clouds = np.vstack((all_pc_data.point_clouds, pc_data.point_clouds))

# load default training parameters
#   why don't we rotate the models??
train_params = default_train_params()

# make ae and folder
encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
train_dir = create_dir(osp.join(top_out_dir, experiment_name))

# save config
conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = train_params['z_rotate'],
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args
        )
conf.experiment_name = experiment_name
conf.held_out_step = 5   # How often to evaluate/print out loss on 
                        # held_out data (if they are provided in ae.train() ).
conf.save(osp.join(train_dir, 'configuration'))

# load config from saved model
load_pre_trained_ae = False
restore_epoch = 500  # whatever epoch you paused at, probably
if load_pre_trained_ae:
    conf = Conf.load(train_dir + '/configuration')
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(conf.train_dir, epoch=restore_epoch)

# build ae model
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)

# train the ae model
buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
train_stats = ae.train(all_pc_data, conf, log_file=fout)
fout.close()