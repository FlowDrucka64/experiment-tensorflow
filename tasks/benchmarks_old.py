from glob import glob
from invoke import task
from json import loads as json_loads
from os import makedirs
from os import remove
from os.path import join, exists
from pandas import read_csv
from pprint import pprint
from requests import post
from subprocess import run as sp_run
from ratelimiter import RateLimiter
from threading import Thread
from time import time
from multiprocessing import Process
from random import random
import numpy as np
import base64
import json
from subprocess import run, PIPE
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

import requests

import matplotlib.pyplot as plt

BM_ROOT = join(PROJ_ROOT, "benchmarking")


def _init_csv_file(func, rate):
    if "_" in func:
        func = func.split("_")[0] + "+" + func.split("_")[1]
    result_dir = join(BM_ROOT, "data")
    makedirs(result_dir, exist_ok=True)

    csv_name = "bm_{}_{}.csv".format(func, rate)
    csv_file = join(result_dir, csv_name)
    with open(csv_file, "w") as out_file:
        out_file.write(
            "Start,Latency,Scheduled,modelRead,modelBuild,imageRead,tensors,imgResize,interpreterLoops,outputPrep,labelsRead\n"
        )

def _write_csv_line(func, rate, num_run, time_segs):
    if "_" in func:
        func = func.split("_")[0] + "+" + func.split("_")[1]
    result_dir = join(BM_ROOT, "data")
    csv_name = "bm_{}+{}.csv".format(func, rate)
    csv_file = join(result_dir, csv_name)

    line = "{}".format(num_run)

    for seg in time_segs:
        line = line + "," + str(seg)
    line = line + "\n"
    # print(line)
    with open(csv_file, "a") as out_file:
        out_file.write(line)

def _get_times_from_json(result):
    times = []
    times.append(int(result["timestamp"]))

    # Filter out all our timestamps
    datas = result["output_data"].split("\n")
    for i in range(len(datas)):
        if len(datas[i]) > 0:
            if datas[i][0] == "[":
                times.append(int(datas[i].split("]")[1].split(" ")[1]))

    # Calc duration of each timestamp
    durations = []
    durations.append(int(result["finished"] - int(result["timestamp"])))
    for i in range(1, len(times)):
        durations.append(times[i] - times[i - 1])

    # pprint(durations)
    return durations

def new_do_single_run(msg_ids, url, msg):
    response = post(
        url, json=msg, timeout=None, headers={"Connection": "close"}
    )

    # Get the async message id
    if response.status_code != 200:
        print(
            "Initial request failed: {}:\n{}".format(
                response.status_code, response.text
            )
        )
    msg_ids.add(int(response.text.strip()))

def do_single_run(func, modelsize, msg_ids):
    """
    Invoke func asnchronously once and add the returned id to msg_ids
    """
    host, port = get_faasm_invoke_host_port()
    url = "http://{}:{}".format(host, port)
    msg = {
        "user": "tf",
        "function": "tf_" + func,
        "cmdline": "{}".format(modelsize),
        "async": True,
    }
    response = post(url, json=msg, timeout=None)

    # Get the async message id
    if response.status_code != 200:
        print(
            "Initial request failed: {}:\n{}".format(
                response.status_code, response.text
            )
        )
    msg_ids.add(int(response.text.strip()))

def poll(func, msg_ids, rate, rep):
    host, port = get_faasm_invoke_host_port()
    url = "http://{}:{}".format(host, port)
    poll_interval = 0.1
    while len(msg_ids) != 0:
        for msg_id in msg_ids:
            status_msg = {
                "user": "tf",
                "function": "tf_" + func,
                "status": True,
                "id": msg_id,
            }
            # print("Posting to {} msg:".format(url))
            # pprint(status_msg)
            response = post(url, json=status_msg)
            # print("Response: {}".format(response.text))

            if response.text.startswith("RUNNING"):
                sleep(poll_interval)
                continue
            elif response.text.startswith("FAILED"):
                print("WARNING: Call failed!")
                msg_ids.remove(msg_id)
                break
            elif not response.text:
                print("WARNING: Empty response")
                msg_ids.remove(msg_id)
                break
            else:
                # First, get the result from the response text
                result_json = json_loads(response.text)
                _write_csv_line(
                    func, rate, rep, _get_times_from_json(result_json)
                )

                # If we reach this point it means the call has succeeded
                msg_ids.remove(msg_id)
                break

        # print("Waiting for {} more respones".format(len(msg_ids)))

def new_stagger_requests(func, msize, msg_ids, duration, rate):

    host, port = get_faasm_invoke_host_port()
    url = "http://{}:{}".format(host, port)
    msg = {
        "user": "tf",
        "function": "tf_" + func,
        "cmdline": "{}".format(msize),
        "async": True,
    }
    for _ in range(duration):
        begin = time()
        for _ in range(int(rate)):
            new_do_single_run(msg_ids, url, msg)
        delay = time() - begin

        if (1 - delay) > 0:
            sleep(1 - delay)

def newbenchmarkRunner(duration, rates, runs, func, msizes):
    print("---------------------------------------------")
    for msize in msizes:
        for rate in rates:
            _init_csv_file(func + "+" + str(msize), str(rate))
            print(
                "RUNNING BENCHMARK FOR {}_{} with {} req/s for {} runs ".format(
                    func, msize, rate, runs
                )
            )
            for run in range(runs):
                print("STARTING RUN #{}".format(run))
                flush_hosts()
                msg_ids = set()
                threads = []
                start = time()
                cnt = 0
                for _ in range(10):
                    t = Thread(
                        target=new_stagger_requests,
                        args=(func, msize, msg_ids, duration, rate / 10),
                    )
                    t.daemon = True
                    t.start()
                    threads.append(t)
                    cnt += int(duration * rate / 10)

                print("Waiting for threads to finish")
                print("Threads: ")
                pprint(threads)
                for t in threads:
                    t.join()
                stop = time()
                print("Planned to send {} req/s".format(rate))
                print(
                    "Sent {} requests in {} seconds: {} req/s".format(
                        cnt, stop - start, cnt / (stop - start)
                    )
                )

                print("Polling results")
                poll(func, msg_ids, str(msize) + "_" + str(rate), run)
                print("---------------------------------------------")
                cnt = 0

def stagger_requests(func, msize, msg_ids, rate):
    for _ in range(int(rate)):
        print("doing single run")
        do_single_run(func, msize, msg_ids)


def benchmarkRunner(duration, rates, runs, func, msizes):
    print("---------------------------------------------")
    for msize in msizes:
        for rate in rates:
            _init_csv_file(func + "+" + str(msize), str(rate))
            print(
                "RUNNING BENCHMARK FOR {}_{} with {} req/s for {} runs ".format(
                    func, msize, rate, runs
                )
            )
            for run in range(runs):
                print("STARTING RUN #{}".format(run))
                flush_hosts()
                msg_ids = set()
                threads = []
                start = time()
                cnt = 0
                for _ in range(duration):
                    begin = time()
                    t = Thread(
                        target=stagger_requests,
                        args=(func, msize, msg_ids, rate),
                    )
                    t.daemon = True
                    t.start()
                    threads.append(t)
                    cnt += int(rate)
                    delay = time() - begin
                    # print(delay)
                    if (1 - delay) > 0:
                        sleep(1 - delay)

                print("Waiting for threads to finish")
                for t in threads:
                    t.join()
                stop = time()
                print("Planned to send {} req/s".format(rate))
                print(
                    "Sent {} requests in {} seconds: {} req/s".format(
                        cnt, stop - start, cnt / (stop - start)
                    )
                )
                print("Polling results")
                poll(func, msg_ids, str(msize) + "_" + str(rate), run)
                print("---------------------------------------------")
                cnt = 0


@task
def throughput(ctx, func, msizes):
    if func == "all":
        targets = ["image", "image_zygote", "image_nostate"]
    else:
        targets = [func]

    if msizes == "all":
        msizes = ["41", "150", "317", "569"]
    else:
        msizes = [msizes]
    rates = [10, 20, 50, 100, 200]

    runs = 3
    requests = 1000
    requests = 200

    print("RUNNING BENCHMARKS FOR {}".format(targets))
    for target in targets:
        for msize in msizes:
            for rate in rates:
                print(
                    "Starting {} runs for {}-{}-{}".format(
                        runs, target, msize, rate
                    )
                )
                vegeta(ctx, target, msize, rate, requests, runs)


@task
def flush(ctx):
    flush_hosts()


@task
def single_run(ctx, func, msize):
    msg_ids = set()
    do_single_run(func, msize, msg_ids)
    host, port = get_faasm_invoke_host_port()
    url = "http://{}:{}".format(host, port)
    while len(msg_ids) > 0:
        for msg_id in msg_ids:
            status_msg = {
                "user": "tf",
                "function": "tf_" + func,
                "status": True,
                "id": msg_id,
            }
            # print("Posting to {} msg:".format(url))
            pprint(status_msg)
            response = post(url, json=status_msg)
            if response.text.startswith("RUNNING"):
                sleep(1)
            elif response.text.startswith("FAILED"):
                print("WARNING: Call failed!")
                msg_ids.remove(msg_id)
                break
            elif not response.text:
                print("WARNING: Empty response")
                msg_ids.remove(msg_id)
                break
            else:
                # First, get the result from the response text
                result_json = json_loads(response.text)
                # print(result_json)
                pprint(_get_times_from_json(result_json))
                msg_ids.remove(msg_id)
                break


def _load_results(dir, segname):
    result_dict = {}
    result_dir = join(BM_ROOT, "data", dir)
    for csv in glob(join(result_dir, "bm_*.csv")):
        workload = csv.split("_")[1]
        np = csv.split("_")[2].split(".")[0]
        df = read_csv(csv)
        if workload not in result_dict:
            result_dict[workload] = {}
        result_dict[workload][np] = [
            df[segname].mean(),
            df[segname].sem(),
        ]
    pprint(result_dict)
    return result_dict


def _my_load_results(dir, segments):
    result_dict = {}
    result_dir = join(BM_ROOT, "data", dir)
    for csv in glob(join(result_dir, "bm_*.csv")):
        workload = csv.split("_")[1]
        np = csv.split("_")[2].split(".")[0]
        df = read_csv(csv)
        for seg in segments:
            if workload not in result_dict:
                result_dict[workload] = {}
            if seg not in result_dict[workload]:
                result_dict[workload][seg] = {}
            result_dict[workload][seg][np] = [
                df[seg].mean(),
                df[seg].sem(),
            ]

    pprint(result_dict)


@task
def plot(ctx, segnames, dir):

    if segnames == "all":
        segnames = [
            "TimeSec",
            "scheduled",
            "modelRead",
            "modelBuild",
            "imageRead",
            "tensors",
            "imgResize",
            "interpreterLoops",
            "outputPrep",
            "labelsRead",
        ]
    else:
        segnames = [segnames]

    for seg in segnames:
        results = _load_results(dir, seg)
        plot_dir = join(BM_ROOT, "plot")
        makedirs(plot_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        for ind, workload in enumerate(results):
            label = workload
            label = label.replace("image", "faasm")
            label = label.replace("+", " ")
            xs = list(results[workload].keys())
            xs.sort(key=int)
            ys = [results[workload][x][0] for x in xs]
            ys_err = [results[workload][x][1] for x in xs]
            ax.errorbar(
                [int(x) for x in xs],
                ys,
                yerr=ys_err,
                linestyle=TLESS_LINE_STYLES[ind],
                marker=".",
                label="{}".format(label),
                color=TLESS_PLOT_COLORS[ind],
            )

        ax.legend()
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Requests per second")
        ax.set_ylabel("Average {} Time [ms]".format(seg))
        # fig.tight_layout()
        fig.suptitle(label.split(" ")[1] + " " + seg + " Benchmark")
        figname = "benchmark_{}.jpg".format(seg)
        plt.savefig(join(plot_dir, figname), format="jpg")


@task
def barplot(ctx, dir):
    segnames = [
        "scheduled",
        "modelRead",
        "modelBuild",
        "imageRead",
        "tensors",
        "imgResize",
        "outputPrep",
        "labelsRead",
    ]
    segnames = ["scheduled", "modelRead", "modelBuild"]
    plot_dir = join(BM_ROOT, "plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    results = _load_results(dir, "scheduled")
    bottom = np.zeros(len(results[list(results.keys())[0]]))
    for seg in segnames:
        print(seg)
        results = _load_results(dir, seg)
        print("SEG: {}".format(seg))
        pprint(results)
        for ind, workload in enumerate(results):
            label = workload
            label = label.replace("image", "faasm")
            label = label.replace("+", " ")
            xs = list(results[workload].keys())
            xs.sort(key=int)
            ys = [results[workload][x][0] for x in xs]
            pprint(ys)
            p = ax.bar(xs, ys, 0.5, label=seg, bottom=bottom)
            bottom = bottom + ys
    ax.legend()
    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=0, top=9)
    ax.set_xlabel("Requests per second")
    ax.set_ylabel("Average exec. time [ms]".format(seg))
    plt.suptitle(list(results.keys())[0].replace("+", " ") + " Benchmark")
    # fig.tight_layout()
    figname = "barplot_{}.jpg".format(list(results.keys())[0])
    plt.savefig(join(plot_dir, figname), format="jpg")
    print("{} saved.".format(figname))


@task
def table(ctx, dir):
    segments = [
        "scheduled",
        "modelRead",
        "modelBuild",
        "imageRead",
        "tensors",
        "imgResize",
        "outputPrep",
        "labelsRead",
    ]
    # segments = ["scheduled","modelRead","modelBuild"]
    _my_load_results(dir, segments)
