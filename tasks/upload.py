from invoke import task
from os.path import exists, join
from requests import put
from tasks.util.faasm import fetch_latest_wasm, get_faasm_upload_host_port
from tasks.util.env import (
    PROJ_ROOT,
    TF_DATA_FILES,
    TF_FUNCTIONS,
    TF_STATE_FILES,
)

@task(default=True)
def all(ctx):
    wasm(ctx)
    data(ctx)
    state(ctx)

@task()
def wasm(ctx, user_in=None, fetch=False):
    """
    Upload the Webassembly files for the TF Lite benchmarks. You can fetch the latest version
    (built in the faasm/examples repo) using sudo and --fetch
    """
    host, port = get_faasm_upload_host_port()
    for f in TF_FUNCTIONS:
        if user_in:
            user = f[0]
        else:
            user = f[0]
        func = f[1]
        if fetch:
            fetch_latest_wasm(user, func)

        wasm_file = join(PROJ_ROOT, "wasm", user, func, func + ".wasm")
        if not exists(wasm_file):
            print("Can not find wasm file: {}".format(wasm_file))
            print("Consider running with `--fetch`: `inv upload.wasm --fetch`")
            raise RuntimeError("WASM function not found")
        url = "http://{}:{}/f/{}/{}".format(host, port, user, func)
        print("Putting function to {}".format(url))
        response = put(url, data=open(wasm_file, "rb"))
        print("Response {}: {}".format(response.status_code, response.text))


@task
def data(ctx):
    """
    Upload the auxiliary data files for the TFLite benchmark runs
    """
    host, port = get_faasm_upload_host_port()
    url = "http://{}:{}/file".format(host, port)

    for df in TF_DATA_FILES:
        host_path = df[0]
        faasm_path = df[1]

        if not exists(host_path):
            print("Did not find data at {}".format(host_path))
            raise RuntimeError("Did not find data file")

        print(
            "Uploading TF data ({}) to {} ({})".format(
                host_path, url, faasm_path
            )
        )
        response = put(
            url,
            data=open(host_path, "rb"),
            headers={"FilePath": faasm_path},
        )

        print("Response {}: {}".format(response.status_code, response.text))


@task
def state(ctx, host=None):
    """
    Upload Tensorflow lite state (model)
    """
    host, port = get_faasm_upload_host_port()

    for df in TF_STATE_FILES:
        host_path = df[0]
        user = df[1]
        key = df[2]

        if not exists(host_path):
            print("Did not find data at {}".format(host_path))
            raise RuntimeError("Did not find data file")
        url = "http://{}:{}/s/{}/{}".format(host, port, user, key)
        print("Uploading TF state ({}) to {} ({})".format(host_path, url, key))
        response = put(
            url,
            data=open(host_path, "rb"),
        )

        print("Response {}: {}".format(response.status_code, response.text))
