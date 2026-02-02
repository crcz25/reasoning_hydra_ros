# This message contains a stamped arbitrary vector of feature data

std_msgs/Header header                           # source header where the feature was extracted from
semantic_inference_msgs/FeatureVector[] feature    # underlying feature vector
int16[] choice
uint64[] ids
string[] labels
string[] prompt  # prompt associated with the feature vectors
uint16[] num_observations  # number of observations for each feature vector
