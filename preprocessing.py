root = './sorted_data'

# Load training data
total_data = scipy.io.loadmat(root + 'T.mat')
train_data = total_data['data']
train_label = total_data['label']

# Transpose and expand dimensions
train_data = np.transpose(train_data, (2, 1, 0))
train_data = np.expand_dims(train_data, axis=1)
train_label = np.transpose(train_label)

# Keep only samples with labels 0 and 1
# mask = (train_label[0] == 2) | (train_label[0] == 1)
# train_data = train_data[mask]
# train_label = train_label[:, mask]

# Shuffle the data
shuffle_num = np.random.permutation(len(train_data))
train_data = train_data[shuffle_num, :, :, :]
train_label = train_label[:, shuffle_num]

# Load test data
test_tmp = scipy.io.loadmat(root + 'E.mat')
test_data = test_tmp['data']
test_label = test_tmp['label']

# Transpose and expand dimensions
test_data = np.transpose(test_data, (2, 1, 0))
test_data = np.expand_dims(test_data, axis=1)
test_label = np.transpose(test_label)

# Keep only samples with labels 0 and 1
# mask = (test_label[0] == 2) | (test_label[0] == 1)
# test_data = test_data[mask]
# test_label = test_label[:, mask]

# Standardize data
target_mean = np.mean(train_data)
target_std = np.std(train_data)
train_data = (train_data - target_mean) / target_std
test_data = (test_data - target_mean) / target_std

print("The shape of training data is", train_data.shape)
print("The shape of test data is", test_data.shape)
print("The shape of train label is", train_label.shape)
print("The shape of test label is", test_label.shape)
test_label= test_label-1
test_label
train_label= train_label-1
