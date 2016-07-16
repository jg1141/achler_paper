import numpy as np
def vectorize(instance, exp, exp_labels):
    instance_vector = np.zeros(exp.shape[1])
    for key in instance.keys():
        # add key to labels, if needed
        if not key in exp_labels:
            exp_labels.append(key)
        key_offset = exp_labels.index(key)
        instance_vector.put(key_offset, instance[key])
    return np.matrix(instance_vector), exp_labels

def learn_instance(instance_list, exp, exp_labels):
    for instance in instance_list:
        instance_vector, exp_labels = vectorize(instance, exp, exp_labels)
        # find first 0 row
        for i in range(exp.shape[0]):
            if not exp[i].any():
                exp[i] = instance_vector
                break
    return exp, exp_labels

def learn_property(instance_list, exp, exp_labels):
    for instance in instance_list:
        instance_vector, exp_labels = vectorize(instance, exp, exp_labels)
        # find first row with 1 unmatched element
        for i in range(exp.shape[0]):
            if not exp[i].any():
                break
            else:
                difference = instance_vector - exp[i]
                if np.count_nonzero(difference) == 1:
                    difference_index = np.nonzero(difference)[1][0]
                    exp.put(i * exp.shape[1] + difference_index, difference.item((0, difference_index)))
                break   
    return exp, exp_labels

def softmax(x):
    # output vector which sums to 1 with elements in range 0 ... 1
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out