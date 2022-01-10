import time

timeStartRun = 0
timeStartFPS = 0


def getFPS():

    global timeStartFPS
    durning = time.perf_counter() - timeStartFPS
    if not durning == 0:
        fps = 1.0 / (time.perf_counter() - timeStartFPS)
    else:
        fps = 0.0
    timeStartFPS = time.perf_counter()
    return fps


def resetTimer():
    global timeStartRun
    timeStartRun = time.perf_counter()


def getRunTime():
    global timeStartRun
    print(time.perf_counter() - timeStartRun)
    timeStartRun = time.perf_counter()
