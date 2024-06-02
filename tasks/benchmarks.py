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


@task
def do_single_run(ctx, func):
    host, port = get_faasm_invoke_host_port()
    modelsize = 569
    url = "http://{}:{}".format(host, port)
    msg = {
        "user": "tf",
        "function": "tf_" + func,
        "cmdline": "{}".format(modelsize),
    }
    response = post(url, json=msg, timeout=None)
    print(response.text)


@task
def throughput_native(ctx, msizes):
    """
    Run the throughput experiment (func,msizes)
    """

    if msizes == "all":
        msizes = ["41", "150", "317", "569"]
    else:
        msizes = [msizes]

    runs = 3
    requests = 1000
    rates = [10, 20, 50, 100, 200]

    # TODO: remove for actual runs
    runs = 1
    requests = 1000
    rates = [10, 20, 50, 100, 200]

    for msize in msizes:
        for rate in rates:
            print("Starting {} run(s) for {}-{}".format(runs, msize, rate))
            vegata_native(ctx, msize, rate, requests, runs)


@task
def throughput_knative(ctx, msizes):
    """
    Run the throughput experiment (func,msizes)
    """

    if msizes == "all":
        msizes = ["41", "150", "317", "569"]
    else:
        msizes = [msizes]

    runs = 3
    requests = 1000
    rates = [10, 20, 50, 100, 200]

    # TODO: remove for actual runs
    runs = 1
    requests = 1000
    rates = [10, 20, 50, 100, 200]

    for msize in msizes:
        for rate in rates:
            print("Starting {} run(s) for {}-{}".format(runs, msize, rate))
            vegata_knative(ctx, msize, rate, requests, runs)


@task
def throughput(ctx, func, msizes):
    """
    Run the throughput experiment (func,msizes)
    """

    if func == "all":
        targets = ["image_state", "image_zygote", "image_nostate"]
    else:
        targets = [func]

    if msizes == "all":
        msizes = ["41", "150", "317", "569"]
    else:
        msizes = [msizes]

    runs = 3
    requests = 1000
    rates = [10, 20, 50, 100, 200]

    # TODO: remove for actual runs
    runs = 1
    requests = 1000
    rates = [10, 20, 50, 100, 200]

    print("RUNNING BENCHMARKS FOR {}".format(targets))
    for target in targets:
        for msize in msizes:
            for rate in rates:
                print(
                    "Starting {} run(s) for {}-{}-{}".format(
                        runs, target, msize, rate
                    )
                )
                vegeta(ctx, target, msize, rate, requests, runs)


def _b64EncodeString(msg):
    msg = json.dumps(msg)
    msg_bytes = msg.encode("ascii")
    base64_bytes = base64.b64encode(msg_bytes)
    return base64_bytes.decode("ascii")


def pre_process_csv_k8(file_path):
    with open(
        file_path.split(".csv")[0] + "_decoded.csv",
        "w",
    ) as out_file:
        out_file.write(
            "Start,Status,Latency,ByteIn,ByteOut,Error,Scheduled,modelRead,modelBuild,imageRead,tensors,imgResize,interpreterLoops,outputPrep,Name,SquenceNumber\n"
        )
        in_file = open(file_path)
        line = in_file.readline()
        while line != "":
            timings = json.loads(base64.b64decode(line.split(",")[6]))[
                "timings"
            ]
            newline = ""
            for i in range(6):
                newline += line.split(",")[i] + ","
            for timing in timings:
                newline += str(timing) + ","
            newline += ",1"
            out_file.write(newline + "\n")
            line = in_file.readline()


def pre_process_csv(file_path):
    with open(
        file_path.split(".csv")[0] + "_decoded.csv",
        "w",
    ) as out_file:

        out_file.write(
            "Start,Status,Latency,ByteIn,ByteOut,Error,Scheduled,modelRead,modelBuild,tensors,imageRead,imgResize,interpreterLoops,outputPrep,Name,SquenceNumber\n"
        )
        in_file = open(file_path)
        line = in_file.readline()
        while line != "":
            newline = ""
            for ind, seg in enumerate(line.split(",")):
                if ind == 6:
                    for ts in (
                        base64.b64decode(seg).decode("ascii").split("\n")
                    ):
                        print(ts)
                        if "[" in ts:
                            newline = newline + "," + ts.split("] ")[1]
                else:
                    newline = newline + "," + seg

            newline = newline[1:]
            out_file.write(newline)
            line = in_file.readline()


@task
def vegata_native(ctx, msize, rate, requests, runs):
    duration = int(int(requests) / int(rate))
    result_dir = join(BM_ROOT, "data", "native")
    makedirs(result_dir, exist_ok=True)
    csv_name = "{}_{}.csv".format(msize, rate)
    csv_file = join(result_dir, csv_name)
    if exists(csv_file):
        remove(csv_file)

    url = "http://172.17.16.1:8080"
    payload = "GET " + url
    for r in range(int(runs)):
        print("Starting run {}/{}".format(r + 1, runs))
        cmd = [
            "echo ",
            json.dumps(payload),
            " |",
            "vegeta attack",
            "-rate={}".format(rate),
            "-duration={}s".format(duration),
            "-connections=1000",
            "-keepalive=false",
            "-timeout=6000s",
            "|",
            "vegeta encode",
            "-to {}".format("csv"),
            " >> {}".format(csv_file),
        ]
        cmd = " ".join(cmd)
        print(cmd)
        run(cmd, check=True, shell=True, cwd=PROJ_ROOT)
    print("Preprocessing....")
    print(csv_file)
    pre_process_csv_k8(csv_file)
    print("Done preprocessing :)")


@task
def vegata_knative(ctx, msize, rate, requests, runs):
    duration = int(int(requests) / int(rate))
    result_dir = join(BM_ROOT, "data", "knative")
    makedirs(result_dir, exist_ok=True)
    csv_name = "{}_{}.csv".format(msize, rate)
    csv_file = join(result_dir, csv_name)
    if exists(csv_file):
        remove(csv_file)

    url = "http://flow-predict.default.127.0.0.1.sslip.io"
    payload = "GET " + url
    for r in range(int(runs)):
        print("Starting run {}/{}".format(r + 1, runs))
        cmd = [
            "echo ",
            json.dumps(payload),
            " |",
            "vegeta attack",
            "-rate={}".format(rate),
            "-duration={}s".format(duration),
            "-connections=1000",
            "-keepalive=false",
            "-timeout=6000s",
            "|",
            "vegeta encode",
            "-to {}".format("csv"),
            " >> {}".format(csv_file),
        ]
        cmd = " ".join(cmd)
        print(cmd)
        run(cmd, check=True, shell=True, cwd=PROJ_ROOT)
    print("Preprocessing....")
    print(csv_file)
    pre_process_csv_k8(csv_file)
    print("Done preprocessing :)")


@task
def vegeta(ctx, func, msize, rate, requests, runs):
    duration = int(int(requests) / int(rate))
    result_dir = join(BM_ROOT, "data", func)
    makedirs(result_dir, exist_ok=True)
    csv_name = "{}_{}.csv".format(msize, rate)
    csv_file = join(result_dir, csv_name)
    if exists(csv_file):
        remove(csv_file)

    host, port = get_faasm_invoke_host_port()
    url = "http://{}:{}".format(host, port)
    payload = {
        "method": "post",
        "url": url,
        "body": _b64EncodeString(
            {
                "user": "tf",
                "function": "tf_" + func,
                "cmdline": "{}".format(msize),
            }
        ),
    }
    for r in range(int(runs)):
        print("Starting run {}/{}".format(r + 1, runs))
        flush_hosts()
        cmd = [
            "echo '",
            json.dumps(payload),
            "' |",
            "vegeta attack",
            "-format=json",
            "-rate={}".format(rate),
            "-duration={}s".format(duration),
            "-connections=1000",
            "-keepalive=false",
            "-timeout=6000s",
            "|",
            "vegeta encode",
            "-to {}".format("csv"),
            " >> {}".format(csv_file),
        ]
        cmd = " ".join(cmd)
        print(cmd)
        run(cmd, check=True, shell=True, cwd=PROJ_ROOT)
    print("Preprocessing....")
    pre_process_csv(csv_file)
    print("Done preprocessing :)")


# --------------------------------------------------------------------------------------------------------------
# ------------------------------Data processing here (plots,statistics)-----------------------------------------
# --------------------------------------------------------------------------------------------------------------
