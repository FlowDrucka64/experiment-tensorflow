from glob import glob
from invoke import task
from json import loads as json_loads
from os import makedirs
from os import remove
from os.path import join, exists, basename
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

plot_segs = [
    "Latency",
    "Scheduled",
    "modelRead",
    "modelBuild",
    "imageRead",
    "tensors",
    "imgResize",
    "interpreterLoops",
    "outputPrep",
]

functions = [
    "image_state",
    "image_zygote",
    "image_nostate",
    #"knative",
    #"native",
]

BM_ROOT = join(PROJ_ROOT, "benchmarking")

# --------------------------------------------------------------------------------------------------------------
# ------------------------------Data processing here (plots,statistics)-----------------------------------------
# --------------------------------------------------------------------------------------------------------------


@task()
def box_plot_collage(ctx, seg, msi):
    if msi == "all":
        msizes = ["41", "150", "317", "569"]
    else:
        msizes = [msi]
    plt.figure(figsize=(12, 5))
    plot_funcs = functions
    # plot_funcs = ["image_nostate"]
    rates = [10, 20, 50, 100, 200]

    for modelsize in msizes:
        plot_data = {}
        for func in plot_funcs:
            if func not in plot_data:
                plot_data[func] = {}

            result_dir = join(BM_ROOT, "data", func)
            for csv in glob(join(result_dir, "*_decoded.csv")):
                msize, rate = (
                    basename(csv).split("_")[0],
                    basename(csv).split("_")[1],
                )
                if msize == modelsize:
                    print("Using data from " + csv)
                    plot_data[func][rate] = list(load_csv(csv, True)[seg])

        real_plot_data = {}
        for func in plot_data:
            if func not in real_plot_data:
                real_plot_data[func] = []
            for rate in rates:
                real_plot_data[func].append(plot_data[func][str(rate)])
                for ind in range(len(real_plot_data[func])):
                    for x in range(len(real_plot_data[func][ind])):
                        real_plot_data[func][ind][x] = (
                            real_plot_data[func][ind][x] / 1000000
                        )

        x_values = rates

        pos = [[], [], []]
        pos[0] = np.arange(len(x_values)) * 1.5
        pos[1] = [p + 0.5 for p in pos[0]]
        pos[2] = [p - 0.5 for p in pos[0]]

        print(pos[0])
        print(pos[1])
        print(pos[2])

        for ind, func in enumerate(plot_funcs):
            plt.boxplot(
                real_plot_data[func], positions=pos[ind], showfliers=False
            )

        plt.xlabel("Requests per second")

        plt.ylabel("Average exec. time [ms]".format(seg))

        plt.title("Boxplot of {}-{}".format(seg, modelsize))

        plt.xticks([i * 1.5 for i in range(len(x_values))], x_values)

        figname = "boxplot_{}_{}.jpg".format(seg, modelsize)
        plot_dir = join(BM_ROOT, "plot", "boxplots")
        plt.savefig(join(plot_dir, figname), format="jpg")
        print(figname + " saved.")
        plt.clf()


@task()
def line_plot_compare(ctx, seg):
    plot_dir = join(BM_ROOT, "plot", "compare", "lineplot")
    makedirs(plot_dir, exist_ok=True)

    msize = "569"

    fig, ax = plt.subplots(figsize=(8, 4))
    for ind, func in enumerate(functions):
        results = load_data(func)
        label = "{} ".format(func)
        xs = list(results[msize].keys())
        xs.sort(key=int)
        ys = [results[msize][x][seg][0] / 1000000 for x in xs]
        ys_err = [results[msize][x][seg][1] / 1000000 for x in xs]
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
    # ax.set_ylim(bottom=0)
    ax.set_xlabel("Requests per second")
    ax.set_ylabel("Average {} Time [ms]".format(seg))
    # fig.tight_layout()
    fig.suptitle("{} Benchmark".format(seg))
    figname = "lineplot_{}_{}.jpg".format(func, seg)
    plt.savefig(join(plot_dir, figname), format="jpg")
    print(figname + " saved.")


@task()
def line_plot_collage(ctx, seg):
    if seg == "all":
        segs = plot_segs
    else:
        segs = [seg]

    for seg in segs:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        plot_dir = join(BM_ROOT, "plot", "collages", "lineplots")
        makedirs(plot_dir, exist_ok=True)

        for f_ind, ax in enumerate(axes):
            results = load_data(functions[f_ind])
            for ind, msize in enumerate(results):
                label = "{} MMACS".format(msize)
                xs = list(results[msize].keys())
                xs.sort(key=int)
                ys = [results[msize][x][seg][0] / 1000000 for x in xs]
                ys_err = [results[msize][x][seg][1] / 1000000 for x in xs]
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
            ax.set_xlabel("Requests per second")
            ax.set_ylabel("Average {} Time [ms]".format(seg))
            ax.set_title(functions[f_ind])
            ax.label_outer()
        # fig.tight_layout()
        fig.suptitle("{} Benchmark collage".format(seg))
        figname = "lineplot_collage_{}.jpg".format(seg)
        plt.savefig(join(plot_dir, figname), format="jpg")
        print(figname + " saved.")


@task()
def bar_plot_collage(ctx, func):
    if func == "all":
        plot_funcs = functions
    else:
        plot_funcs = [func]

    msizes = ["41", "150", "317", "569"]

    for func in plot_funcs:
        fig, axes = plt.subplots(1, 5, figsize=(12, 4), sharey=True)
        plot_dir = join(BM_ROOT, "plot", "collages", "lineplots")
        makedirs(plot_dir, exist_ok=True)
        results = load_data(functions[f_ind])
        for f_ind, ax in enumerate(axes):
            for ind, msize in enumerate(results):
                pass


@task()
def bar_plot(ctx, func, msize):
    if func == "all":
        plot_funcs = functions
    else:
        plot_funcs = [func]

    if msize == "all":
        msizes = ["41", "150", "317", "569"]
    else:
        msizes = [msize]

    segs = [
        #"Scheduled",
        "modelRead",
        "modelBuild",
        #"imageRead",
        "tensors",
        #"imgResize",
        #"interpreterLoops",
        "outputPrep",
    ]
    for func in plot_funcs:
        plot_dir = join(BM_ROOT, "plot", func, "barplot")
        makedirs(plot_dir, exist_ok=True)
        results = load_data(func)
        for msize in msizes:
            bottom = np.zeros(5)
            fig, ax = plt.subplots(figsize=(8, 6))
            for seg in segs:
                xs = list(results[msize].keys())
                xs.sort(key=int)
                ys = [results[msize][x][seg][0] / 1000000 for x in xs]
                ax.bar(xs, ys, 0.5, label=seg, bottom=bottom)
                bottom = bottom + ys
            ax.legend()
            ax.set_xlim(left=-0.5)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Requests per second")
            ax.set_ylabel("Average {} time [ms]".format(func))
            fig.suptitle("{} {} Benchmark".format(func, msize))
            # fig.tight_layout()

            figname = "barplot_{}_{}.jpg".format(func, msize)
            plt.savefig(join(plot_dir, figname), format="jpg")
            print(figname + " saved.")


@task()
def line_plot(ctx, func):
    if func == "all":
        plot_funcs = functions
    else:
        plot_funcs = [func]

    segs = plot_segs

    for func in plot_funcs:
        plot_dir = join(BM_ROOT, "plot", func, "lineplot")
        makedirs(plot_dir, exist_ok=True)
        results = load_data(func)

        for seg in segs:
            fig, ax = plt.subplots(figsize=(8, 4))
            for ind, msize in enumerate(results):
                label = "{} MMACS".format(msize)
                xs = list(results[msize].keys())
                xs.sort(key=int)
                ys = [results[msize][x][seg][0] / 1000000 for x in xs]
                ys_err = [results[msize][x][seg][1] / 1000000 for x in xs]
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
            # ax.set_ylim(bottom=0)
            ax.set_xlabel("Requests per second")
            ax.set_ylabel("Average {} Time [ms]".format(seg))
            # fig.tight_layout()
            fig.suptitle("{} {} Benchmark".format(func, seg))
            figname = "lineplot_{}_{}.jpg".format(func, seg)
            plt.savefig(join(plot_dir, figname), format="jpg")
            print(figname + " saved.")


def load_data(function):
    result_dict = {}
    result_dir = join(BM_ROOT, "data", function)
    for csv in glob(join(result_dir, "*_decoded.csv")):
        msize, rate = basename(csv).split("_")[0], basename(csv).split("_")[1]
        if msize not in result_dict:
            result_dict[msize] = {}
        result_dict[msize][rate] = load_csv(csv)
    # pprint(result_dict)
    return result_dict


#
# Loads a single csv file into a panda df and computes the
# segment times from the timestamps
#
def load_csv(filepath, return_df=False):
    in_df = read_csv(filepath)
    out_df = in_df.copy(deep=True)
    segs = [
        "Start",
        "Scheduled",
        "modelRead",
        "modelBuild",
        "imageRead",
        "tensors",
        "imgResize",
        "interpreterLoops",
        "outputPrep",
    ]
    for ind, seg in enumerate(segs):
        if ind != 0:
            out_df[seg] = in_df[seg] - in_df[segs[ind - 1]]
    out = {}
    segs = plot_segs
    for seg in segs:
        # pprint(out_df)
        out[seg] = [
            int(out_df[seg].mean()),
            int(out_df[seg].sem()),
        ]  # casting to int since nanosecs are preciese enough
    if return_df:
        return out_df
    # pprint(out)
    return out
