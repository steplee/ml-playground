import numpy as np
import caffe2.python.model_helper as model_helper
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

import pdb

# Returns pred net.
def make(model='zoo_resnet50', create_dist_net=False, standardize=False,normalize=False,img_size=224,**kw):
    input_blob = 'data'
    remove_layers_type = ['FC', 'Softmax'] # By type.
    remove_layers_with = [] # By name.

    assert(model == 'zoo_resnet50') # TODO add the others. If statements to set other params...

    #model = 'zoo_densenet121'
    #output_blob = 'pool5'
    #global_pool = False

    #model = 'zoo_inception2'
    #output_blob = 'pool5/7x7_s1'
    #global_pool = False

    model = 'zoo_resnet50'
    output_blob = 'res4_5_branch2c_bn'
    #output_blob2 = 'res5_2_branch2c_bn'
    pooling = 'global'
    #pooling = 6
    #remove_layers_with = ['res5', 'pool5']

    DIR = '/home/slee/stuff/terrapixel/cpp_localization/'
    model = DIR + 'saves/raw/' + model

    dev = core.DeviceOption(caffe2_pb2.CUDA,0)
    #dev = core.DeviceOption(caffe2_pb2.CPU,0)
    with core.DeviceScope(dev):
        path = '%s/predict_net.pb'%model
        pred_net = caffe2_pb2.NetDef()
        with open(path, 'rb') as f:
            pred_net.ParseFromString(f.read())
        pred_net.device_option.CopyFrom(dev) # GPU
        print('pred do',pred_net.device_option)


        path = '%s/init_net.pb'%model
        init_net = caffe2_pb2.NetDef()
        with open(path, 'rb') as f: init_net.ParseFromString(f.read())
        init_net.device_option.CopyFrom(dev) # GPU

        # Do 3 things:
        #   1) Set device on all ops to gpu
        #   2) if blobs have 'gpu_0' prefix, remove it.
        #   3) Remove FC layers, softmaxes, etc.
        for net in (pred_net, init_net):
            remove_ops = []
            for j,op in enumerate(net.op):
                if op.type in remove_layers_type or any(any(n in out for out in op.output) for n in remove_layers_with):
                    remove_ops.append(j)
                op.device_option.CopyFrom(dev)
                for i,f in enumerate(op.input):
                    op.input[i] = f.replace('gpu_0/','')
                for i,f in enumerate(op.output):
                    op.output[i] = f.replace('gpu_0/','')
            for i,f in enumerate(net.external_input):
                net.external_input[i] = f.replace('gpu_0/','')
            for i,f in enumerate(net.external_output):
                net.external_output[i] = f.replace('gpu_0/','')

            dbg_removed = []
            for rop in remove_ops[::-1]:
                for out in net.op[rop].output:
                    for j,exout in enumerate(net.external_output):
                        if exout == out:
                            net.external_output.pop(j)
                            break
                net.op.pop(rop)
                dbg_removed += [out]
            print('Removed Ops:', ' '.join(dbg_removed))

        pred_net = core.Net(pred_net)
        init_net = core.Net(init_net)

        # Flatten!
        '''
        if pooling == 'global':
            pred_net.AveragePool2D([output_blob], ['output_code_1'], global_pooling=True)
            pred_net.AveragePool2D([output_blob2], ['output_code_2'], global_pooling=True)
            pred_net.Concat(['output_code_1','output_code_2'], ['output_code_','whocares'], axis=1)
            #pred_net.MaxPool2D([output_blob], ['output_code_'], global_pooling=True)
            pred_net.Flatten(['output_code_'], ['output_code'])
        else:
            pred_net.AveragePool2D([output_blob], ['output_code_'], kernel=pooling, stride=pooling)
            pred_net.Flatten(['output_code_'], ['output_code'])
            #pred_net.Copy([output_blob], ['output_code'])
        output_blob = 'output_code'
        '''

        pred_net._net.external_output.append(output_blob)

        mh = model_helper.ModelHelper()
        mh.net = pred_net
        mh.param_init_net = init_net
        mh.RunAllOnGPU()


    #pdb.set_trace()

    #print('model proto', mh.net.Proto())

    # Determine code_size. Reset workspace afterwards.
    SZ = img_size
    img_size = (SZ,SZ)
    fake_x = np.random.randn(*[1, 3, *img_size]).astype(np.float32)
    #print('running param init.')
    workspace.RunNetOnce(init_net)
    #print('running net.')
    workspace.FeedBlob(input_blob, fake_x, device_option=dev)
    workspace.CreateNet(pred_net)
    #workspace.RunNet(pred_net)
    #workspace.RunNetOnce(pred_net)
    #print('fetching.')
    #fake_code = workspace.FetchBlob(output_blob)
    #workspace.ResetWorkspace()
    #workspace.FeedBlob(input_blob, fake_x, device_option=dev)

    #print(fake_code.shape)
    #assert((len(fake_code.shape) == 2 or all(fake_code.shape[2:]==1)) and 'code_size must be 1 dim')
    #code_size = fake_code.shape[1]


    return mh

#make()
