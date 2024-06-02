from os.path import dirname, expanduser, realpath, join
from socket import gethostname

PROJ_ROOT = dirname(dirname(dirname(realpath(__file__))))
FAASM_ROOT_AZ_VM = join(expanduser("~"), "code", "faasm")
FAASM_ROOT = join(expanduser("~"), "faasm")

EXAMPLE_ROOT_AZ_VM = join(expanduser("~"), "code", "examples")
EXAMPLE_ROOT = join(expanduser("~"), "examples")


BIN_DIR = join(PROJ_ROOT, "bin")

SGX_INSTALL_DIR = "/opt/intel"

# Note that this is copied from faasm/experiment-base/tasks/util/env.py
AZURE_RESOURCE_GROUP = "faasm"

# SGX VM config

AZURE_SGX_VM_SIZE = "Standard_DC2ds_v3"
AZURE_SGX_LOCATION = "eastus2"
AZURE_SGX_VM_NAME = "faasm-sgx-vm"
AZURE_SGX_VM_IMAGE = "Canonical:UbuntuServer:18_04-lts-gen2:18.04.202109180"
AZURE_SGX_VM_ADMIN_USERNAME = "faasm"
AZURE_SGX_VM_SSH_KEY_FILE = "{}/experiments/sgx/pkeys".format(PROJ_ROOT)

# Attestation config

AZURE_ATTESTATION_PROVIDER_NAME = "faasmattprov"
AZURE_ATTESTATION_TYPE = "SGX-IntelSDK"

# ----------
# Plot aesthetics
# ----------

TLESS_PLOT_COLORS = [
    (255 / 255, 162 / 255, 0 / 255),
    (62 / 255, 0 / 255, 161 / 255),
    (161 / 255, 0 / 255, 62 / 255),
    (0 / 255, 69 / 255, 22 / 255),
    (0 / 255, 0 / 255, 255 / 255),
]

TLESS_LINE_STYLES = ["-", "--", "-.", ":", "-"]

TLESS_HATCH_STYLES = ["--", "+", "x", "\\", "+"]

# ----------
# TLess Stuff
# ----------

TF_FUNCTIONS = [
    ["tf", "tf_image_state"],
    #["tf", "tf_image_zygote_41"],
    ["tf", "tf_image_zygote_150"],
    #["tf", "tf_image_zygote_new_317"],
    #["tf", "tf_image_zygote_new_569"],
    ["tf", "tf_image_nostate"],
]

# Path for TLess data in this repository
TF_DATA_DIR = join(PROJ_ROOT, "data")

# Path for TLess data to be uplodad in Faasm's filesystem
TF_FAASM_DATA_DIR = "/tflite"

# For each data file, we have the origin path (where we copy data from), and
# the path in Faasm's filesystem we are gonna store the piece of data
TF_DATA_FILES = [
    [
        join(TF_DATA_DIR, "grace_hopper.bmp"),
        join(TF_FAASM_DATA_DIR, "grace_hopper.bmp"),
    ],
    [
        join(TF_DATA_DIR, "labels.txt"),
        join(TF_FAASM_DATA_DIR, "labels.txt"),
    ],
    # [
    #     join(TF_DATA_DIR, "mobilenet_v1_1.0_224.tflite"),
    #     join(TF_FAASM_DATA_DIR, "mobilenet_v1"),
    # ],
    # [
    #     join(TF_DATA_DIR,"models", "mobilenet_v1_0.25_224_41_quant.tflite"),
    #     join(TF_FAASM_DATA_DIR, "mobilenet_v1"),
    # ],
    [
        join(TF_DATA_DIR,"models", "mobilenet_v1_0.5_224_150_quant.tflite"),
        join(TF_FAASM_DATA_DIR, "mobilenet_v1"),
    ],
    # [
    #     join(TF_DATA_DIR,"models", "mobilenet_v1_0.75_224_317_quant.tflite"),
    #     join(TF_FAASM_DATA_DIR, "mobilenet_v1"),
    # ],
    # [
    #     join(TF_DATA_DIR,"models", "mobilenet_v1_1.0_224_569_quant.tflite"),
    #     join(TF_FAASM_DATA_DIR, "mobilenet_v1"),
    # ],
]

TF_STATE_FILES = [
    [
        join(TF_DATA_DIR, "mobilenet_v1_1.0_224.tflite"),
        "tf",
        "mobilenet_v1",
    ],
    [
        join(TF_DATA_DIR, "models", "mobilenet_v1_0.5_224_150_quant.tflite"),
        "tf",
        "mobilenet_v1_150",
    ],
    [
        join(TF_DATA_DIR, "models", "mobilenet_v1_0.25_224_41_quant.tflite"),
        "tf",
        "mobilenet_v1_41",
    ],
    [
        join(TF_DATA_DIR, "models", "mobilenet_v1_0.75_224_317_quant.tflite"),
        "tf",
        "mobilenet_v1_317",
    ],
    [
        join(TF_DATA_DIR, "models", "mobilenet_v1_1.0_224_569_quant.tflite"),
        "tf",
        "mobilenet_v1_569",
    ],
]


def get_faasm_root():
    if "koala" in gethostname():
        return FAASM_ROOT
    else:
        return FAASM_ROOT_AZ_VM


def get_example_root():
    if "koala" in gethostname():
        return EXAMPLE_ROOT
    else:
        return EXAMPLE_ROOT_AZ_VM
