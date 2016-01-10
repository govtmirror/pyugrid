import functools
import logging

from pyugrid.flexible_mesh.mpi import MPI_RANK

level = logging.ERROR
# level = logging.DEBUG

log = logging.getLogger('pyugrid')
log.parent = None
formatter = logging.Formatter(fmt='[%(asctime)s] %(levelname)s: %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

console = logging.StreamHandler()
console.setLevel(level)
console.setFormatter(formatter)
log.addHandler(console)

# fh = logging.FileHandler('/tmp/pyugrid.log', mode='w')
# fh.setFormatter(formatter)
# fh.setLevel(level)
# log.addHandler(fh)


class log_entry_exit(object):

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        log.debug("entering {0} (rank={1})".format(self.f.__name__, MPI_RANK))
        try:
            return self.f(*args, **kwargs)
        finally:
            log.debug("exited {0} (rank={1})".format(self.f.__name__, MPI_RANK))

    def __get__(self, obj, _):
        """Support instance methods."""

        return functools.partial(self.__call__, obj)

