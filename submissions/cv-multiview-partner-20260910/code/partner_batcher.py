"""多案件并发发起复核，由单个模型所有者合并为真实批量，避免多份权重。"""
from concurrent.futures import Future
from queue import Queue, Empty
from threading import Thread
import time


class BatchReviewer:
    def __init__(self, grader, batch_size=3, wait_seconds=.03):
        if type(batch_size) is not int or not 1<=batch_size<=8 or not 0<=wait_seconds<=.2:
            raise ValueError('伙伴批量参数非法')
        self.grader=grader;self.batch_size=batch_size;self.wait_seconds=wait_seconds
        self.queue=Queue();self.closed=False
        self.thread=Thread(target=self._work,name='partner-model-owner',daemon=True);self.thread.start()

    def __call__(self,messages,attempt):
        if self.closed:raise RuntimeError('伙伴模型批处理器已关闭')
        future=Future();self.queue.put((messages,attempt,future));return future.result()

    def _work(self):
        while True:
            first=self.queue.get()
            if first is None:return
            batch=[first];deadline=time.monotonic()+self.wait_seconds
            while len(batch)<self.batch_size:
                try:item=self.queue.get(timeout=max(0,deadline-time.monotonic()))
                except Empty:break
                if item is None:
                    self.queue.put(None);break
                batch.append(item)
            try:
                values=self.grader.generate_batch([(m,a) for m,a,_ in batch])
                if len(values)!=len(batch):raise RuntimeError('伙伴批处理输出数量错误')
                for (_,_,future),value in zip(batch,values):future.set_result(value)
            except BaseException as exc:
                for _,_,future in batch:future.set_exception(exc)

    def close(self):
        self.closed=True;self.queue.put(None);self.thread.join()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
