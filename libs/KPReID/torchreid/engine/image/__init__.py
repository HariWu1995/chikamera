from __future__ import absolute_import

from .softmax import ImageSoftmaxEngine
from .triplet import ImageTripletEngine

# from ...utils.profiling import profile_single_gpu, GigaValue

# gpu_profile = profile_single_gpu(device_id=0)
# gpu_free_memory = round(gpu_profile[-1] / GigaValue, 2)

# if gpu_free_memory < 40:
#     from .part_lowvram_engine import ImagePartBasedEngine
# else:
from .part_based_engine import ImagePartBasedEngine
