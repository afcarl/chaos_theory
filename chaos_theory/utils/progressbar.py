import sys
from .colors import *

def progress_itr(itr, bar_len=20, outstream=sys.stdout, flush_every=1, colors=True):
    try:
        tot_len = len(itr)
    except TypeError:
        raise ValueError("Iterable does not have length!")

    for i, item in enumerate(itr):
        frac = float(i+1)/tot_len
        numeq = min(int(round(bar_len*frac)), bar_len)
        numblank = bar_len-numeq
        prgstr = '\r [%s] (%.2f%%:%d/%d)' % ( ('='*numeq+' '*numblank), frac*100, i+1, tot_len)

        if i == tot_len-1:
            color = 'green'
        elif i >= (tot_len*0.5):
            color = 'yellow'
        else:
            color = 'white'

        if colors:
            print_color(prgstr, fore=color, outstream=outstream)
        else:
            outstream.write(prgstr)

        if i%flush_every == 0:
            outstream.flush()
        yield item
    outstream.write('\n')
    outstream.flush()

