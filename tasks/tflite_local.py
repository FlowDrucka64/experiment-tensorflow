# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

import argparse
from time import time
from invoke import task
from threading import Thread
import numpy as np
from PIL import Image
from time import sleep
from glob import glob
from invoke import task
from json import loads as json_loads
from os import makedirs
from os.path import join
from pandas import read_csv
from pprint import pprint
from requests import post
from requests import get
from subprocess import run as sp_run
from ratelimiter import RateLimiter
from threading import Thread
from time import time
from multiprocessing import Process
from random import random
from tasks.util.faasm import (
    flush_hosts,
    get_faasm_exec_time_from_json,
    get_faasm_invoke_host_port,
)
from tasks.util.env import (
    PROJ_ROOT,
    TLESS_LINE_STYLES,
    TLESS_PLOT_COLORS,
    get_faasm_root,
)
from time import sleep

import matplotlib.pyplot as plt


BM_ROOT = join(PROJ_ROOT, "benchmarking")
NUM_NODES = 7
NUM_CORES_PER_NODE = 4

data = []


def _init_csv_file(m, np,cs):
    result_dir = join(BM_ROOT, "data")
    makedirs(result_dir, exist_ok=True)

    csv_name = "bm_{}-{}_{}.csv".format(m,cs, np)
    csv_file = join(result_dir, csv_name)
    with open(csv_file, "w") as out_file:
        out_file.write("NumRun,TimeSec\n")


def _write_csv_line(m, np, num_run, cs, time_sec):
    result_dir = join(BM_ROOT, "data")
    csv_name = "bm_{}-{}_{}.csv".format(m,cs,np)
    csv_file = join(result_dir, csv_name)
    with open(csv_file, "a") as out_file:
        out_file.write("{},{}\n".format(num_run, time_sec))




def do_single_run(cnt):
    host = "helloworld-python.default.127.0.0.1.sslip.io"
    port = 80
    url = "http://{}:{}".format(host, port)
    start = time()
    for _ in range(cnt):
        resp = get(url)
        #pprint(resp)
    end = time()
    data.append(((end-start)*1000/cnt))

@task
def throughput(ctx):
    cnt = 0
    duration = 10
    rates = [10,20,50,100,200]
    #rates = [10,20]
    for rate in rates:
        _init_csv_file("native", rate, 0)
        threads = []
        start = time()
        for _ in range(duration):
            begin = time()
            t = Thread(target=do_single_run, args=([rate]))
            t.daemon = True
            t.start()
            threads.append(t)
            cnt += int(rate)
            delay = time()-begin
            #print(delay)
            if (1-delay)>0:
                sleep(1-delay)
        stop = time()
        print("Planned to send {} req/s".format(rate))
        print("Sent {} requests in {} seconds: {} req/s".format(cnt,stop-start,cnt/(stop-start)))
        
        print("Waiting for threads to finish")
        for t in threads:
            t.join()  
        print("Polling results")
        #poll(func,msg_ids,rate,0,0)
        print("---------------------------------------------")
        cnt = 0
        print("Results:")
        for d in data:
            print(d)
            _write_csv_line("native",rate,0,0,d)
            data.remove(d)
