from functools import wraps
import logging
import time

logger = logging.getLogger(__name__)


def with_logging(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t_start = time.time()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise e
        finally:
            logger.info(f"Time taken for {func.__name__}: {time.time() - t_start:.2f}s")

    return wrapper
