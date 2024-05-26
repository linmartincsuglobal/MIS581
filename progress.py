
import sys


def n_of_m(n, m):
    num_length = len(str(m))
    sys.stdout.write('\r%0*d of %0*d' % (num_length, n, num_length, m))
    sys.stdout.flush()
