from __future__ import print_function
import os
import sys
import tensorflow as tf
import pandas as pd
from  sklearn.model_selection import train_test_split

dataX = pd.read_csv('./kddcup.data_10_percent/X_1.csv')
datay = pd.read_csv('./kddcup.data_10_percent/y_1.csv')

data_X = dataX.astype('float32')
data_y = datay.astype('float32')

X_train, X_val, y_train, y_val = train_test_split(data_X.values,data_y.values,test_size = 0.2, random_state=42)

tf.app.flags.DEFINE_integer('training_iteration', 100, 'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version nummber of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp/model', 'working directory.')
FLAGS = tf.app.flags.FLAGS
def main(_):
  if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    print('Usage: mnist_saved_model.py [--training_iteration=x] '
          '[--model_version=y] export_dir')
    sys.exit(-1)
  if FLAGS.training_iteration <= 0:
    print('Please specify a positive value for training iteration.')
    sys.exit(-1)
  if FLAGS.model_version <= 0:
    print('Please specify a positive value for version number.')
    sys.exit(-1)


print('Training model...')

sess = tf.InteractiveSession()

serizalized_tf_example = tf.placeholder(tf.string, name='tf_example')
feature_configs = {'x': tf.FixedLenFeature(shape=[28], dtype=tf.float32), }
tf_example = tf.parse_example(serizalized_tf_example, feature_configs)

x = tf.identity(tf_example['x'], name='x')
y_ = tf.placeholder(tf.float32, [None, 5])

w_1 = tf.Variable(tf.truncated_normal([28, 64], stddev=0.1))
b_1 = tf.Variable(tf.constant(0.1, shape=[64]))
h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

w_2 = tf.Variable(tf.truncated_normal([64, 32], stddev=0.1))
b_2 = tf.Variable(tf.constant(0.1, shape=[32]))
h_2 = tf.nn.relu(tf.matmul(h_1, w_2) + b_2)

w_3 = tf.Variable(tf.truncated_normal([32, 5], stddev=0.1))
b_3 = tf.Variable(tf.constant(0.1, shape=[5]))
y = tf.nn.softmax(tf.matmul(h_2, w_3) + b_3, name='y')

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)
values, indices = tf.nn.top_k(y, 5)
table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in range(5)]))
predicition_classes = table.lookup(tf.to_int64(indices))

tf.global_variables_initializer().run()

for _ in range(FLAGS.training_iteration):
    train_step.run(feed_dict={x: X_train, y_: y_train})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print('training accuracy %g' % sess.run(
    accuracy, feed_dict={
        x: X_val,
        y_: y_val
    }))
print('Done training!')


export_path_base = "/tmp/model"
export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes(str(FLAGS.model_version)))
print("Exporting trained model to", export_path)

builder = tf.saved_model.builder.SavedModelBuilder(export_path)

tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs = {'x':tensor_info_x},
        outputs = {'scores': tensor_info_y},
        method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

builder.add_meta_graph_and_variables(
    sess,[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'kdd_predict':
            prediction_signature,
        },
    main_op = tf.tables_initializer(),
    strip_default_attrs=True)

builder.save()

print('Done exporting model for online prediction!')

if __name__ == '__main__':
    tf.app.run()
