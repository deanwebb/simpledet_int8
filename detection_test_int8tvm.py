import tvm
import tvm.relay as relay
from tvm.contrib import graph_runtime
import numpy as np
import cv2


def extract_tvm_tasks(mx_sym, shape, arg_params, aux_params):
    sym, params = relay.frontend.from_mxnet(mx_sym, shape, arg_params=arg_params, aux_params=aux_params)
    with relay.quantize.qconfig(skip_k_conv=0):
        sym = relay.quantize.quantize(sym, params=params)

    target = tvm.target.create('cuda -model=1080ti')

    import tvm.autotvm as autotvm
    tasks = autotvm.task.extract_from_program(sym, target=target, params=params, ops=(relay.nn.deformable_conv2d,))
                                              #ops=(relay.nn.conv2d, relay.nn.dense, relay.nn.conv2d_transpose))
    for i in range(len(tasks)):
        print(tasks[i].name)
        if tasks[i].name == 'topi_nn_dense':
            if tasks[i].args[1][1][0] % 4 == 0 and tasks[i].args[1][1][1] % 4 == 0:
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args, tasks[i].target,
                                          tasks[i].target_host, 'int8')
                tasks[i] = tsk

        elif tasks[i].name == 'topi_nn_conv2d':
            if (tasks[i].workload[2][0] % 4 == 0 and tasks[i].workload[2][1] % 4 == 0) \
                    or 'group' in tasks[i].name:
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args, tasks[i].target,
                                          tasks[i].target_host, 'int8')
                tasks[i] = tsk
        print(tasks[i])
    import pickle
    task_dump = 'tasks/with_reid-int8-{}x{}x{}-{}_deformable_conv2d.pkl'.format(shape['data'][0], shape['data'][2], shape['data'][3], '1080ti')
    with open(task_dump, 'wb') as f:
        pickle.dump(tasks, f)


class TVMTester(object):
    def __init__(self, mx_sym, shape, arg_params, aux_params):
        self.batch_size = shape['data'][0]
        self.im_shape = shape['data'][2:]
        sym, params = relay.frontend.from_mxnet(mx_sym, shape, arg_params=arg_params, aux_params=aux_params)
        with relay.quantize.qconfig(skip_k_conv=0):
            sym = relay.quantize.quantize(sym, params=params)
        sym = relay.ir_pass.infer_type(sym)

        target = tvm.target.create('cuda -model=1080ti')

        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(sym, params=params, target=target)
        self.mod = graph_runtime.create(graph, lib, tvm.gpu(0))
        self.mod.set_input(**params)

        self.ctx = tvm.gpu(0)


    def preprocessing(self, ims):
        im_infos = [None] * self.batch_size
        height, width = self.im_shape
        self.im_info = np.array([height, width, 1], dtype=np.float32)

        for i, im in enumerate(ims):
            ims[i] = cv2.resize(im, self.im_shape, interpolation=cv2.INTER_LINEAR)
            im_infos[i] = self.im_info
        data_batch = np.array(ims, dtype=np.float32)
        data_batch = np.transpose(data_batch, (0, 3, 2, 1))
        im_infos = np.array(im_infos)
        data_batch = {'data': tvm.nd.array(data_batch, ctx=self.ctx), 'im_info': tvm.nd.array(im_infos, ctx=self.ctx)}
        return data_batch


    def test_batch(self, ims, unused):
        data_batch = self.preprocessing(ims)
        #output = self.predict(data_batch)
        #results = self.postprocessing(output)
        self.mod.set_input(**data_batch)
        for i in range(5): # repeat a few times
            self.mod.run()


