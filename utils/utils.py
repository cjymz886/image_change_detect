import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def (func):
    """计时器装饰器，用于记录函数执行时间"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        logger.debug(f"开始执行函数: {func_name}")

        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"函数 {func_name} 执行完成，耗时: {elapsed_time:.3f}秒")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"函数 {func_name} 执行失败，耗时: {elapsed_time:.3f}秒，错误: {str(e)}")
            raise

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper