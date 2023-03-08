from .util.switch import Switch

disable_xformers = Switch()
disable_voodoo = Switch()
enable_local_ray_temp = Switch()
disable_progress = Switch()
disable_download_progress = Switch()
enable_ray_alternative = Switch()


class InvalidModelException(Exception):
    pass


class InvalidModelCacheException(InvalidModelException):
    pass
