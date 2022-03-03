import os
import datetime
import configparser
import tensorflow as tf

from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from model import MultiLabelClassifier
from dataset import LoadData

flags.DEFINE_string('batch_size', '32', 'batch size')
flags.DEFINE_string('epochs', '10', 'epochs')
flags.DEFINE_string('learning_rate', '0.001', 'learning rate')
flags.DEFINE_string('gpu', '0', 'gpu')
flags.DEFINE_string('dataset_id', '0', 'dataset id')

def modify_config(cfg):
    """
    This function modifies the config.
    """
    cfg['train']['batch_size'] = FLAGS.batch_size
    cfg['train']['epochs'] = FLAGS.epochs
    cfg['train']['learning_rate'] = FLAGS.learning_rate
    return cfg

def main(_):
    """
    This function is the main function.
    """
    # Set the GPU to use
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    # Set the batch size
    batch_size = int(FLAGS.batch_size)
    # Set the epochs
    epochs = int(FLAGS.epochs)
    # Set the learning rate
    learning_rate = float(FLAGS.learning_rate)
    # Set the dataset id
    dataset_id = FLAGS.dataset_id
    # Change config file
    curpath = os.path.dirname(os.path.realpath(__file__))
    cfgpath = os.path.join(curpath, "configs", "configs.ini")
    configs = configparser.ConfigParser()
    configs.read(cfgpath, encoding="utf-8")
    # import data
    dataset_path = os.path.join(curpath, "data", dataset_id)
    dataset_loader = LoadData(os.path.join(dataset_path, "images"), 
                                os.path.join(dataset_path, "labels", "list_attr_celeba.csv"),
                                int(configs['train']['input_size']))
    train_ds, val_ds, num_classes = dataset_loader.load_data()
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    # Set model
    model = MultiLabelClassifier(configs, num_classes, training=True)

    cfg = modify_config(configs)
    # Set Checkpoint callback
    checkpoint_path = "./checkpoint/cp-{epoch:04d}.ckpt"
    checkpoint = ModelCheckpoint(checkpoint_path,
                                verbose=1,
                                save_weights_only=True,
                                save_freq=5000)
    # Set tensorboard callback function
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.CosineDecay(learning_rate, 3*len(train_ds)))
    # Compile the model
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.BinaryAccuracy()],
              run_eagerly=True)
    # Model summary
    model.summary()
    # Train the model
    history = model.fit(
        train_ds,   # training data
        validation_data=val_ds,   # validation data
        epochs=epochs,   # number of epochs
        verbose=1,   # verbosity
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5), checkpoint, tensorboard_callback]
    )
    # Save the history
    with open(os.path.join(curpath, "history", "history.txt"), "w") as f:
        f.write(str(history.history))
    
    # Save the model
    model_save = MultiLabelClassifier(configs, num_classes, training=False)
    model_save.load_weights('checkpoint/cp-0006.ckpt')
    model_save.save(os.path.join(curpath, "savedmodel"))

if __name__ == '__main__':
    app.run(main)