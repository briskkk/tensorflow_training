import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from __future__ import print_function
import os
import sys

attack_map = {'back.':'4',
              'buffer_overflow.':'1',
              'ftp_write.':'2',
              'guess_passwd.':'2',
              'imap.':'2',
              'ipsweep.':'3',
              'land.':'4',
              'loadmodule.':'1',
              'multihop.':'2',
              'neptune.':'4',
              'nmap.':'3',
              'perl.':'1',
              'phf.':'2',
              'pod.':'4',
              'portsweep.':'3',
              'rootkit.':'1',
              'satan.':'3',
              'smurf.':'4',
              'spy.':'2',
              'teardrop.':'4',
              'warezclient.':'2',
              'warezmaster.':'2',
              'normal.':'0',
              'unknown':'unknown'}

service_map ={'aol':'0',
          'auth':'1',
          'bgp':'2',
          'courier':'3',
          'csnet_ns':'4',
          'ctf':'5',
          'daytime':'6',
          'discard':'7',
          'domain':'8',
          'domain_u':'9',
          'echo':'10',
          'eco_i':'11',
          'ecr_i':'12' ,
          'efs':'13',
          'exec':'14',
          'finger':'15',
          'ftp':'16',
          'ftp_data':'17',
          'gopher':'18',
          'harvest':'19',
          'hostnames':'20',
          'http':'21',
          'http_2784':'22',
          'http_443':'23',
          'http_8001':'24',
          'imap4':'25',
          'IRC':'26',
          'iso_tsap':'27',
          'klogin':'28',
          'kshell':'29',
          'ldap':'30',
          'link':'31',
          'login':'32',
          'mtp':'33',
          'name':'34',
          'netbios_dgm':'35',
          'netbios_ns':'36',
          'netbios_ssn':'37',
          'netstat':'38',
          'nnsp':'39',
          'nntp':'40',
          'ntp_u':'41',
          'other':'42',
          'pm_dump':'43',
          'pop_2':'44',
          'pop_3':'45',
          'printer':'46',
          'private':'47',
          'red_i':'48',
          'remote_job':'49',
          'rje':'50',
          'shell':'51',
          'smtp':'52',
          'sql_net':'53',
          'ssh':'54',
          'sunrpc':'55',
          'supdup':'56',
          'systat':'57',
          'telnet':'58',
          'tftp_u':'59',
          'tim_i':'60',
          'time':'61',
          'urh_i':'62',
          'urp_i':'63',
          'uucp':'64',
          'uucp_path':'65',
          'vmnet':'66',
          'whois':'67',
          'X11':'68',
          'Z39_50':'69'}


flag_map = {'OTH':'0',
            'REJ':'1',
            'RSTO':'2',
            'RSTOS0':'3',
            'RSTR':'4',
            'S0':'5',
            'S1':'6',
            'S2':'7',
            'S3':'8',
            'SF':'9',
            'SH':'10'}
protocol_map = {'tcp':'1',
                'udp':'2',
                'icmp':'0'}

def transform(kdd):
    kdd[41] = kdd[41].replace(attack_map)
    kdd[2] = kdd[2].replace(service_map)
    kdd[3] = kdd[3].replace(flag_map)
    kdd[1] = kdd[1].replace(protocol_map)
    return kdd

def savememory(kdd):
    for i in range(9):
        kdd[i] = kdd[i].astype(np.float32)
    for i in range(22,42):
        kdd[i] = kdd[i].astype(np.float32)
    return kdd

kdd = pd.read_csv('/home/sinet/trafficdata/kddcup.data', usecols=[0,1,2,3,4,5,6,7,8,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41] , iterator=True, header=None)
loop = True
chunkSize = 100000
chunks = []
index = 0
while loop:
    try:
        chunk = kdd.get_chunk(chunkSize)
        chunk = transform(chunk)
        chunk = savememory(chunk)
        chunks.append(chunk)
        index+=1
    except StopIteration:
        loop = False
        print('Iteration is stopped.')
print('开始合并')
kdd = pd.concat(chunks, ignore_index=True)

kdd1 = kdd.iloc[:,:28]
kdd2 = kdd[41]


encoder = OneHotEncoder()
def onehot(self):
    return encoder.fit_transform(self.values.reshape(-1,1)).toarray()
y = onehot(kdd2)
X = kdd1.values

X_train, X_val,y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=42)

tf.app.flags.DEFINE_integer('training_iteration', 100, 'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 4, 'version nummber of the model.')
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