import sys, math, random
from heapq import heappush, heappop
from bisect import bisect_right, bisect_left
from collections import Counter, deque, defaultdict
from itertools import permutations


MOD = 998244353
MOD = 10**9 + 7
RANDOM = random.randrange(2**62)

def solve():
    n = int(sys.stdin.readline().strip())
    d = defaultdict(list)
    for i in range(n - 1):
        L = list(map(int, sys.stdin.readline().split()))
        d[L[0]].append(L[1])
        d[L[1]].append(L[0])
    
    in1 = [0 for i in range(n)]
    out1 = in1[:]
    dpin = [0 for i in range(n)]

    stack = [(1, -1, 0)]
    while stack:
        cur, prev, phase = stack.pop()
        if phase == 0:
            in1[cur - 1] = 1
            stack.append((cur, prev, 1))
            for i in d[cur]:
                if i == prev:
                    continue
                stack.append((i, cur, 0))
        elif phase == 1:
            for i in d[cur]:
                if i == prev:
                    continue
                in1[cur - 1] += in1[i - 1]
                dpin[cur - 1] += dpin[i - 1]
            dpin[cur - 1] += in1[cur - 1] - 1
            
    stack = [(1, -1, 0)]
    while stack:
        cur, prev, phase = stack.pop()
        if phase == 0:
            if prev != -1:
                out1[cur - 1] = out1[prev - 1] + len(out1) - dpin[cur - 1] + dpin[prev - 1] - 2 * in1[cur - 1]
            stack.append((cur, prev, 1))
            for i in d[cur]:
                if i == prev:
                    continue
                stack.append((i, cur, 0))
        
        elif phase == 1:
            dpin[cur - 1] += out1[cur - 1]

    print(*dpin)

solve()