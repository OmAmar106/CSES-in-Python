import sys,os
from io import BytesIO, IOBase

BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._file = file
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

def solve():
    L = list(map(int, sys.stdin.readline().split()))
    L1 = list(map(int, sys.stdin.readline().split()))
    L2 = []
    L1 = sorted([(L1[i]<<20)+i for i in range(len(L1))])

    for i in range(L[0]):
        if len(L2)>=4 and (L2[-4]>>20)==(L1[i]>>20):
            continue
        else:
            L2.append(L1[i])
    L1 = L2
    for k in range(len(L1)-3):
        for i in range(k+1,len(L1)-2):
            start = i+1
            end = len(L1)-1
            while start<end:
                y = (L1[k]>>20)+(L1[i]>>20)+(L1[start]>>20)+(L1[end]>>20)
                if y>L[1]:
                    end -= 1
                elif y<L[1]:
                    start += 1
                else:
                    print((L1[k]&((1<<20)-1))+1,(L1[i]&((1<<20)-1))+1,(L1[start]&((1<<20))-1)+1,(L1[end]&((1<<20)-1))+1)
                    exit() 
    print("IMPOSSIBLE")
    #L1 = LII()
    #st = SI()
solve()