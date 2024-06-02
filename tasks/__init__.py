from invoke import Collection


from . import benchmarks
from . import upload
from . import tflite_local
from . import plots

ns = Collection(benchmarks, upload, tflite_local, plots)
