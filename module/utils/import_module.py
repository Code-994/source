import importlib
import os
import warnings

def import_modules(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and file.endswith(".py")
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            # 导入失败时跳过该模块（warn 而不是 raise），避免单个模块阻断全部注册
            # importlib.import_module(namespace + "." + model_name)
            try:
                importlib.import_module(namespace + "." + model_name)
            except Exception as e:
                warnings.warn(
                    f"[import_modules] 跳过 {namespace}.{model_name}: {e}"
                )