import tvm
import tvm.relay as relay
from tvm.contrib import graph_runtime
import numpy as np
import cv2

from multiprocessing import Process
import os

from core.detection_module import DetModule
from core.detection_input import Loader
from utils.load_model import load_checkpoint

from six.moves import reduce
from six.moves.queue import Queue
from threading import Thread
import argparse
import importlib
import mxnet as mx
import numpy as np
import six.moves.cPickle as pkl
def calibrate_dataset(batch,shape):
    for i,batch_data in enumerate(batch):
        if(i>20):
            break
        data=batch_data.data[0].asnumpy()
        im_info=batch_data.data[1].asnumpy()
        im_id=batch_data.data[2].asnumpy()
        rec_id=batch_data.data[3].asnumpy()
        data_corped=[]
        data=data.transpose((0,2,3,1))
        for _,im in enumerate(data):
            data_corped.append(cv2.resize(im,shape,interpolation=cv2.INTER_LINEAR))
        data_corped=np.array(data_corped,dtype=np.float32)
        data_corped.transpose((0,3,1,2))
        im_info[0][0]=np.float32(shape[0])
        im_info[0][1]=np.float32(shape[1])
        yield {'data':tvm.nd.array(data_corped,ctx=tvm.gpu(0)),'im_info':tvm.nd.array(im_info,ctx=tvm.gpu(0)),\
               'im_id':tvm.nd.array(im_id,ctx=tvm.gpu(0)),'rec_id':tvm.nd.array(rec_id,ctx=tvm.gpu(0))}

def quantize_mx_sym(mx_sym,shape,arg_params,aux_params,loader_cali):
    shape_im=shape['data'][2:]
    sym, params = relay.frontend.from_mxnet(mx_sym, shape, arg_params=arg_params, aux_params=aux_params)
    with relay.quantize.qconfig(skip_k_conv=0):
        sym=relay.quantize.quantize(sym,params=params,dataset=calibrate_dataset(loader_cali,shape_im))
        #sym=relay.quantize.quantize(sym,params=params,dataset=None)
    sym=relay.ir_pass.infer_type(sym)
    return(sym,params)
        

class TVMTester(object):
    def __init__(self, sym_quant, shape, params,GPU_id):
        self.batch_size = shape['data'][0]
        self.im_shape = shape['data'][2:]


        target = tvm.target.create('cuda -model=1080ti')

        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(sym_quant, params=params, target=target)
        print("build finished")
        self.mod = graph_runtime.create(graph, lib, tvm.gpu(GPU_id))
        print("created")
        self.mod.set_input(**params)

        self.ctx = tvm.gpu(GPU_id)


    def preprocessing(self, batch_data):
        data=batch_data.data[0].asnumpy()
        im_info=batch_data.data[1].asnumpy()
        im_id=batch_data.data[2].asnumpy()
        rec_id=batch_data.data[3].asnumpy()
        data_resized=[]
        data=data.transpose((0,2,3,1))
        for i, im in enumerate(data):
            data_resized.append(cv2.resize(im, self.im_shape, interpolation=cv2.INTER_LINEAR))
        data_resized=np.array(data_resized,dtype=np.float32)
        data_resized=data_resized.transpose((0,3,1,2))

        im_info[0][0] = np.float32(self.im_shape[0])
        im_info[0][1] = np.float32(self.im_shape[1])

        data_batch = {'data': tvm.nd.array(data_resized, ctx=self.ctx), 'im_info': tvm.nd.array(im_info, ctx=self.ctx),\
                      'im_id':tvm.nd.array(im_id, ctx=self.ctx),'rec_id':tvm.nd.array(rec_id, ctx=self.ctx)}
        return data_batch


    def forward(self,batch_data,is_train=False):
        data_batch = self.preprocessing(batch_data)
        self.mod.set_input(**data_batch)
        self.mod.run()
    
    def get_outputs(self):
        num = self.mod.get_num_outputs()
        output=[]
        for i in range(num):
            output.append(self.mod.get_output(i))
        return(output)


def parse_args():
    parser = argparse.ArgumentParser(description='Test Detection')
    # general
    parser.add_argument('--config', help='config file path', type=str)
    args = parser.parse_args()

    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    return config


if __name__ == "__main__":
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    config = parse_args()

    pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
    transform, data_name, label_name, metric_list = config.get_config(is_train=False)

    sym = pModel.test_symbol
    sym.save(pTest.model.prefix + "_test.json")

    image_sets = pDataset.image_set
    image_sets_cali = ("coco_valminusminival2014",)

    roidbs = [pkl.load(open("data/cache/{}.roidb".format(i), "rb"), encoding="latin1") for i in image_sets]
    roidb = reduce(lambda x, y: x + y, roidbs)
    roidb = pTest.process_roidb(roidb)

    roidbs_cali = [pkl.load(open("data/cache/{}.roidb".format(i), "rb"), encoding="latin1") for i in image_sets_cali]
    roidb_cali = reduce(lambda x, y: x + y, roidbs_cali)
    roidb_cali = pTest.process_roidb(roidb_cali)

    for i, x in enumerate(roidb):
        x["rec_id"] = i
    
    for i, x in enumerate(roidb_cali):
        x["rec_id"] = i

    loader = Loader(roidb=roidb,
                    transform=transform,
                    data_name=data_name,
                    label_name=label_name,
                    batch_size=1,
                    shuffle=False,
                    num_worker=8,
                    num_collector=2,
                    worker_queue_depth=4,
                    collector_queue_depth=4,
                    kv=None)
    
    loader_cali = Loader(roidb=roidb_cali,
                         transform=transform,
                         data_name=data_name,
                         label_name=label_name,
                         batch_size=1,
                         shuffle=False,
                         num_worker=4,
                         num_collector=2,
                         worker_queue_depth=2,
                         collector_queue_depth=2,
                         kv=None)
#    loader_cali=None            
    print("total number of images: {}".format(loader.total_record))
#    print("total number of images_cali: {}".format(loader_cali.total_record))

    data_names = [k[0] for k in loader.provide_data]
    
    arg_params, aux_params = load_checkpoint(pTest.model.prefix, pTest.model.epoch)
    shape_={'data':loader.provide_data[0][1],\
            'im_info':loader.provide_data[1][1],\
            'im_id':loader.provide_data[2][1],\
            'rec_id':loader.provide_data[3][1]}
    print(shape_)
    sym_q,params=quantize_mx_sym(sym,shape_,arg_params,aux_params,loader_cali)
    print("symbol quantized")
    execs = []
    for i in pKv.gpus:
        mod=TVMTester(sym_q,shape_,params,i)
        execs.append(mod)
    print("all model build")
    all_outputs = []

    data_queue = Queue(100)
    result_queue = Queue()

    def eval_worker(exe, data_queue, result_queue):
        while True:
            batch = data_queue.get()
            exe.forward(batch, is_train=False)
            out = [x.asnumpy() for x in exe.get_outputs()]
            result_queue.put(out)

    #workers = [Thread(target=eval_worker, args=(exe, data_queue, result_queue)) for exe in execs]
    #for w in workers:
    #    w.daemon = True
    #    w.start()
        
    pworkers = [Process(target=eval_worker, args=(exe, data_queue, result_queue)) for exe in execs]
    for pw in pworkers:
        pw.start()
        pw.join()
    import time
    t1_s = time.time()

    def data_enqueue(loader, data_queue):
        for batch in loader:
            data_queue.put(batch)
    enqueue_worker = Thread(target=data_enqueue, args=(loader, data_queue))
    enqueue_worker.daemon = True
    enqueue_worker.start()

    for _ in range(loader.total_record):
        r = result_queue.get()
        print(result_queue.qsize())

        rid, id, info, cls, box = r
        rid, id, info, cls, box = rid.squeeze(), id.squeeze(), info.squeeze(), cls.squeeze(), box.squeeze()
        # TODO: POTENTIAL BUG, id or rid overflows float32(int23, 16.7M)
        id = np.asscalar(id)
        rid = np.asscalar(rid)

        scale = info[2]  # h_raw, w_raw, scale
        box = box / scale  # scale to original image scale
        cls = cls[:, 1:]   # remove background
        # TODO: the output shape of class_agnostic box is [n, 4], while class_aware box is [n, 4 * (1 + class)]
        box = box[:, 4:] if box.shape[1] != 4 else box

        output_record = dict(
            rec_id=rid,
            im_id=id,
            im_info=info,
            bbox_xyxy=box,  # ndarray (n, class * 4) or (n, 4)
            cls_score=cls   # ndarray (n, class)
        )

        all_outputs.append(output_record)

    t2_s = time.time()
    print("network uses: %.1f" % (t2_s - t1_s))

    # let user process all_outputs
    all_outputs = pTest.process_output(all_outputs, roidb)

    # aggregate results for ensemble and multi-scale test
    output_dict = {}
    for rec in all_outputs:
        im_id = rec["im_id"]
        if im_id not in output_dict:
            output_dict[im_id] = dict(
                bbox_xyxy=[rec["bbox_xyxy"]],
                cls_score=[rec["cls_score"]]
            )
        else:
            output_dict[im_id]["bbox_xyxy"].append(rec["bbox_xyxy"])
            output_dict[im_id]["cls_score"].append(rec["cls_score"])

    for k in output_dict:
        if len(output_dict[k]["bbox_xyxy"]) > 1:
            output_dict[k]["bbox_xyxy"] = np.concatenate(output_dict[k]["bbox_xyxy"])
        else:
            output_dict[k]["bbox_xyxy"] = output_dict[k]["bbox_xyxy"][0]

        if len(output_dict[k]["cls_score"]) > 1:
            output_dict[k]["cls_score"] = np.concatenate(output_dict[k]["cls_score"])
        else:
            output_dict[k]["cls_score"] = output_dict[k]["cls_score"][0]

    t3_s = time.time()
    print("aggregate uses: %.1f" % (t3_s - t2_s))

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO(pTest.coco.annotation)

    if callable(pTest.nms.type):
        nms = pTest.nms.type(pTest.nms.thr)
    else:
        from operator_py.nms import py_nms_wrapper
        nms = py_nms_wrapper(pTest.nms.thr)

    def do_nms(k):
        bbox_xyxy = output_dict[k]["bbox_xyxy"]
        cls_score = output_dict[k]["cls_score"]
        final_dets = {}

        for cid in range(cls_score.shape[1]):
            score = cls_score[:, cid]
            if bbox_xyxy.shape[1] != 4:
                cls_box = bbox_xyxy[:, cid * 4:(cid + 1) * 4]
            else:
                cls_box = bbox_xyxy
            valid_inds = np.where(score > pTest.min_det_score)[0]
            box = cls_box[valid_inds]
            score = score[valid_inds]
            det = np.concatenate((box, score.reshape(-1, 1)), axis=1).astype(np.float32)
            det = nms(det)
            dataset_cid = coco.getCatIds()[cid]
            final_dets[dataset_cid] = det
        output_dict[k]["det_xyxys"] = final_dets
        del output_dict[k]["bbox_xyxy"]
        del output_dict[k]["cls_score"]
        return (k, output_dict[k])

    from multiprocessing import cpu_count
    from multiprocessing.pool import Pool
    pool = Pool(cpu_count())
    output_dict = pool.map(do_nms, output_dict.keys())
    output_dict = dict(output_dict)

    t4_s = time.time()
    print("nms uses: %.1f" % (t4_s - t3_s))

    coco_result = []
    for iid in output_dict:
        result = []
        for cid in output_dict[iid]["det_xyxys"]:
            det = output_dict[iid]["det_xyxys"][cid]
            if det.shape[0] == 0:
                continue
            scores = det[:, -1]
            xs = det[:, 0]
            ys = det[:, 1]
            ws = det[:, 2] - xs + 1
            hs = det[:, 3] - ys + 1
            result += [
                {'image_id': int(iid),
                 'category_id': int(cid),
                 'bbox': [float(xs[k]), float(ys[k]), float(ws[k]), float(hs[k])],
                 'score': float(scores[k])}
                for k in range(det.shape[0])
            ]
        result = sorted(result, key=lambda x: x['score'])[-pTest.max_det_per_image:]
        coco_result += result

    t5_s = time.time()
    print("convert to coco format uses: %.1f" % (t5_s - t4_s))

    import json
    json.dump(coco_result,
              open("experiments/{}/{}_result.json".format(pGen.name, pDataset.image_set[0]), "w"),
              sort_keys=True, indent=2)

    ann_type = 'bbox'
    coco_dt = coco.loadRes(coco_result)
    coco_eval = COCOeval(coco, coco_dt)
    coco_eval.params.useSegm = (ann_type == 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    t6_s = time.time()
    print("coco eval uses: %.1f" % (t6_s - t5_s))
