def conv_layer_visitor(conv_layer):
    
    if type(conv_layer['config']['activation'])==dict:
            activation = conv_layer['config']['activation']['class_name']
    else:
            activation = conv_layer['config']['activation']
    return {
        'class_name': conv_layer['class_name'],
        'filters': conv_layer['config']['filters'],
        'kernel_size': conv_layer['config']['kernel_size'],
        'strides': conv_layer['config']['strides'],
        'padding':  conv_layer['config']['padding'],
        'activation': activation
    }
         

def no_name_layer_visitor(layer):
    return {
        'class_name': layer['class_name'],
        'config': {k: v for k, v in (layer['config']).items() if k != 'name'}
    }
    
visitors = {
    'InputLayer': no_name_layer_visitor,
    'Conv1D': conv_layer_visitor,
    'Conv2D': conv_layer_visitor,
    'Conv2DTranspose': conv_layer_visitor,
    'MaxPooling1D': no_name_layer_visitor,
    'MaxPooling2D': no_name_layer_visitor,
    'GlobalMaxPooling1D': no_name_layer_visitor,
    'Dense': no_name_layer_visitor,
    'Dropout': no_name_layer_visitor,
    'BatchNormalization': no_name_layer_visitor,
    'Reshape': no_name_layer_visitor,
    'Flatten': no_name_layer_visitor
}
    
    
def layer_comparator(expected, actual):
    class_name = expected['class_name']
    visitor = visitors[class_name] if class_name in visitors else None
    
    if visitor is None:
        # This should't happen because we are using the expected layer
        return False  # We don't know how to compare this layer :(
    
    return visitor(expected) == visitor(actual)
 