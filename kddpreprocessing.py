import pandas as pd


kdd = pd.read_csv('/home/briskk/Datasets/kdddd/kddcup.data_10_percent_corrected.csv',names = [ 'duration'
, 'protocol_type'
, 'service'
, 'flag'
, 'src_bytes'
, 'dst_bytes'
, 'land'
, 'wrong_fragment'
, 'urgent'
, 'hot'
, 'num_failed_logins'
, 'logged_in'
, 'num_compromised'
, 'root_shell'
, 'su_attempted'
, 'num_root'
, 'num_file_creations'
, 'num_shells'
, 'num_access_files'
, 'num_outbound_cmds'
, 'is_host_login'
, 'is_guest_login'
, 'count'
, 'srv_count'
, 'serror_rate'
, 'srv_serror_rate'
, 'rerror_rate'
, 'srv_rerror_rate'
, 'same_srv_rate'
, 'diff_srv_rate'
, 'srv_diff_host_rate'
, 'dst_host_count'
, 'dst_host_srv_count'
, 'dst_host_same_srv_rate'
, 'dst_host_diff_srv_rate'
, 'dst_host_same_src_port_rate'
, 'dst_host_srv_diff_host_rate'
, 'dst_host_serror_rate'
, 'dst_host_srv_serror_rate'
, 'dst_host_rerror_rate'
, 'dst_host_srv_rerror_rate'
, 'attack_type' ])


attack_map = {'back.':'dos',
              'buffer_overflow.':'u2r',
              'ftp_write.':'r2l',
              'guess_passwd.':'r2l',
              'imap.':'r2l',
              'ipsweep.':'probe',
              'land.':'dos',
              'loadmodule.':'u2r',
              'multihop.':'r2l',
              'neptune.':'dos',
              'nmap.':'probe',
              'perl.':'u2r',
              'phf.':'r2l',
              'pod.':'dos',
              'portsweep.':'probe',
              'rootkit.':'u2r',
              'satan.':'probe',
              'smurf.':'dos',
              'spy.':'r2l',
              'teardrop.':'dos',
              'warezclient.':'r2l',
              'warezmaster.':'r2l',
              'normal.':'normal',
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


kdd['attack_type'] = kdd['attack_type'].replace(attack_map)
kdd["service"] = kdd["service"].replace(service_map)
kdd['flag'] = kdd['flag'].replace(flag_map)


def getprotocol(protocol_type):
    if protocol_type == "tcp":
        return 1
    elif protocol_type == "udp":
        return 2
    else:
        return 0


kdd["protocol_type"] = kdd["protocol_type"].apply(getprotocol)

at_map={'normal':'0',
        'u2r':'1',
        'r2l':'2',
        'probe':'3',
        'dos':'4'}

kdd['attack_type'] = kdd['attack_type'].replace(at_map)

kdd1 = kdd.iloc[:,:9]
kdd2 = kdd.iloc[:,22:]
kddconcat = pd.concat([kdd1,kdd2],axis=1)


X = pd.DataFrame(kddconcat.iloc[:,:-1],dtype=float)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
def onehot(self):
    return encoder.fit_transform(self.values.reshape(-1,1)).toarray()
y = pd.DataFrame(onehot(kddconcat['attack_type']),dtype=float)

X = X.astype('float32')
y = y.astype('float32')

X.to_csv('/home/briskk/Datasets/kdddd/X_1.csv',index=False)
y.to_csv('/home/briskk/Datasets/kdddd/y_1.csv',index=False)