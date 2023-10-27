# 获取 deepspeed 训练配置
GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        tb_path="",
                        tb_name=""):
    """
    这是一个函数，它接受多个参数，用于配置 DeepSpeed 训练\n
    offload: 是否启用参数离载（parameter offload）。如果为 True，参数会被加载到 CPU，否则为 "none"\n
    stage：这是零优化（Zero Optimization）的阶段，默认值为 2\n
    enable_hybrid_engine：是否启用混合引擎（hybrid engine）\n
    inference_tp_size：用于配置推断的线程池（thread pool）大小，默认值为 1\n
    release_inference_cache：是否释放推断缓存\n
    pin_parameters：是否将参数固定（pin）到内存中\n
    tp_gather_partition_size：用于配置线程池的收集分区大小\n
    max_out_tokens：最大输出token数\n
    enable_tensorboard：是否启用 TensorBoard 日志记录\n
    tb_path：用于指定 TensorBoard 日志文件保存路径的参数\n
    tb_name：用于指定 TensorBoard 日志的名称\n
    """ 
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "bfloat16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }

