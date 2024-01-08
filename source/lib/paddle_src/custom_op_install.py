from paddle.utils import (
    cpp_extension,
)

# NOTE: 请在本文件的目录下执行:python custom_op_install.py install,以安装自定义算子

cpp_extension.setup(
    name="paddle_deepmd_lib",
    ext_modules=cpp_extension.CUDAExtension(
        sources=[
            "../src/coord.cc",
            "../src/env_mat_nvnmd.cc",
            "../src/env_mat.cc",
            "../src/ewald.cc",
            "../src/fmt_nlist.cc",
            "../src/gelu.cc",
            "../src/map_aparam.cc",
            "../src/neighbor_list.cc",
            "../src/pair_tab.cc",
            "../src/prod_env_mat_nvnmd.cc",
            # "../src/prod_env_mat.cc",
            "../src/prod_force_grad.cc",
            "../src/prod_force.cc",
            "../src/prod_virial_grad.cc",
            "../src/prod_virial.cc",
            "../src/region.cc",
            "../src/SimulationRegion.cpp",
            "../src/soft_min_switch_force_grad.cc",
            "../src/soft_min_switch_force.cc",
            "../src/soft_min_switch_virial_grad.cc",
            "../src/soft_min_switch_virial.cc",
            "../src/soft_min_switch.cc",
            "../src/tabulate.cc",
            "../src/utilities.cc",
            "../src/cuda/coord.cu",
            "../src/cuda/gelu.cu",
            "../src/cuda/neighbor_list.cu",
            # "../src/cuda/prod_force_grad.cu",
            # "../src/cuda/prod_force.cu",
            # "../src/cuda/prod_virial_grad.cu",
            # "../src/cuda/prod_virial.cu",
            "../src/cuda/region.cu",
            "../src/cuda/tabulate.cu",
            "./paddle_prod_env_mat.cu",
            "./paddle_prod_env_mat.cc",
            "./paddle_prod_virial_grad.cu",
            "./paddle_prod_virial_grad.cc",
            "./paddle_prod_virial.cu",
            "./paddle_prod_virial.cc",
            "./paddle_prod_force.cu",
            "./paddle_prod_force.cc",
            "./paddle_prod_force_grad.cu",
            "./paddle_prod_force_grad.cc",
            "./paddle_neighbor_stat.cc",
        ],
        include_dirs=[
            "../../lib/include/",
        ],
        library_dirs=["/usr/local/cuda-11/lib64"],
        define_macros=[("GOOGLE_CUDA", "1")],
    ),
)
