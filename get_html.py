import os
import json
from multiprocessing import Process
import csv
import requests
csv.field_size_limit(100000000)
class Worker():
    def __init__(self, src_fp, tgt_fp):
        self.src_fp = src_fp
        self.tgt_fp = tgt_fp
        self.postfix = '.pid.'

    def run(self, pid, p_num):
        pid_file_fp = self.tgt_fp
        with open(self.src_fp, 'r',encoding='utf-8') as f_in:
            for idx, line in enumerate(f_in):
                if idx % p_num != pid: continue
                try:
                    with open(pid_file_fp+'/'+str(idx)+'.html', 'w') as f_out:
                        out_string = requests.get(line.strip()).text
                        if out_string: f_out.write(out_string)
                except:
                    continue

    def merge_result(self, keep_pid_file=False):
        return 
        # os.system('cat %s%s* > %s' % (self.tgt_fp, self.postfix, self.tgt_fp))
        # if not keep_pid_file:
        #     os.system('rm %s%s*' % (self.tgt_fp, self.postfix))

class MultiProcessor():
    def __init__(self, worker, pid_num):
        self.worker = worker
        self.pid_num = pid_num

    def run(self):
        for pid in range(self.pid_num):
            p = Process(target= self.worker.run, args = (pid, self.pid_num))
            p.start()
            
        for pid in range(self.pid_num):
            p.join()

if __name__ == "__main__":
    import json
        
    worker = Worker('msmarco-url.txt','msmarco')
    mp = MultiProcessor(worker, 10)
    mp.run()
    print("All Processes Done.")
    # worker.merge_result(keep_pid_file=True)