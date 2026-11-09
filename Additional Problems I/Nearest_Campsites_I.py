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

NEG = -(1<<31)
MAXI = 2*10**5

def update(bit,i,x):
    i += 1
    while i<=MAXI:
        if x>bit[i]:
            bit[i] = x
        else:
            break
        i += i&-i
def query(bit,i):
    res = NEG
    while i:
        res = max(res,bit[i])
        i -= i&-i
    return res

seg11 = [NEG]*(MAXI+1)
seg12 = seg11[:]
seg13 = seg11[:]
seg14 = seg11[:]

def solve():
    n,m = list(map(int, sys.stdin.readline().split()))

    L = []
    for i in range(n):
        u,v = list(map(int, sys.stdin.readline().split()))
        L.append(((u<<40)+(v<<20)))

    L1 = []
    for i in range(m):
        u,v = list(map(int, sys.stdin.readline().split()))
        L1.append(((u<<40)+(v<<20)+i))

    msk = (1<<20)-1
    d = {}
    for index,value in enumerate(sorted(set((i>>20)&(msk) for i in L+L1))):
        d[value] = index

    L.sort()
    L1.sort()

    ans = [float('inf')]*len(L1)
    i = 0
    for val in L1:
        u,v,index = (val>>40),(val>>20)&msk,val&msk
        while i<len(L) and (L[i]>>40)<=u:
            val = L[i]
            u1,v1 = (val>>40),(val>>20)&msk
            z = d[v1]
            update(seg11,MAXI-1-z,(u1-v1))
            update(seg12,z,u1+v1)
            i += 1

        z = d[v]
        ans[index] = min(u-v-query(seg11,MAXI-z),u+v-query(seg12,z))

    # for j in range(len(seg11.tree)):
    #     seg11.tree[j] = seg12.tree[j] = NEG
    
    i = len(L)-1
    for i1 in range(len(L1)-1,-1,-1):
        val = L1[i1]
        u,v,index = (val>>40),(val>>20)&msk,val&msk
        while i>=0 and (L[i]>>40)>=u:
            val = L[i]
            u1,v1 = (val>>40),(val>>20)&msk
            z = d[v1]
            update(seg13,MAXI-1-z,-u1-v1)
            update(seg14,z,-u1+v1)
            i -= 1
        # print(u,v,u-v,seg11.query(v,MAXI),u+v-seg12.query(0,v))
        z = d[v]
        ans[index] = min(ans[index],-u-v-query(seg13,MAXI-z),-u+v-query(seg14,z))

    print(max(ans))
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = SI()
solve()