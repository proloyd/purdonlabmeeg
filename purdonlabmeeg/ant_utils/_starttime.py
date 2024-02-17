import datetime


def _read_starttime(fname):
    with open(fname, 'rb') as f:
        f.seek(0, 2)
        chartotal = f.tell()
        startdatenum = None
        for ii in range(2000):
            f.seek(chartotal - ii - 1, 0)
            try:
                currtext = f.read(11).decode('utf-8')
                if currtext == '[StartDate]':
                    f.readline().decode('utf-8')
                    startdatenum = float(f.readline().decode('utf-8').strip('\n'))
                    break
            except UnicodeDecodeError as e:
                pass
    if startdatenum is None:
        return startdatenum
    starttime = datetime.datetime.fromordinal(
        int((datetime.datetime(1899, 12, 30, 0, 0, 0).toordinal() + startdatenum) // 1)) +\
              datetime.timedelta(days=startdatenum%1)
    return starttime
