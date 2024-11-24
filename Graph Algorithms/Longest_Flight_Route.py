import sys,math,random
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations

# functions #
MOD = 998244353
MOD = 10**9 + 7
RANDOM = random.randrange(2**62)
def gcd(a,b):
    if a%b==0:
        return b
    else:
        return gcd(b,a%b)
def lcm(a,b):
    return a//gcd(a,b)*b
def w(x):
    return x ^ RANDOM
##

#String hashing : sh, fenwick sortedlist : fsortl, Number : numtheory, SparseTable : SparseTable
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull
#Combinatorics : pnc, Diophantine Equations : dpheq
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

from types import GeneratorType

def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to
    return wrappedfunc

# @bootstrap
# put this just on top of recursion function to increase the recursion limit

# rather than return now use yield and when function being called inside itself, use yield before the function name
# example usage:
# @bootstrap
# def rec1(L,k,cur,count):
# 	if count>=100000:
# 		yield float('INF')
# 	if cur+k+1>=len(L)-1:
# 		yield L[cur]+2
# 	if cur in d:
# 		yield d[cur]
# 	ans = float('INF')
# 	mini = float('INF')
# 	for i in range(k+1,0,-1):
# 		if L[cur+i]<mini:
# 			ans = min(ans,1+L[cur]+(yield rec1(L,k,cur+i,count+1)))
# 			mini = L[cur+i]
# 	d[cur] = ans
# 	yield ans
# the limit of recursion on cf is 10**6

def kahn(graph,n):

    indeg, idx = [0] * n, [0] * n
    for i in range(n):
        for e in graph[i]:
            indeg[e] += 1

    q, res = [], []
    for i in range(n):
        if indeg[i] == 0:
            q.append(i)  # heappush(q, -i)

    nr = 0
    while q:
        res.append(q.pop())  # res.append(-heappop(q))
        idx[res[-1]], nr = nr, nr + 1
        for e in graph[res[-1]]:
            indeg[e] -= 1
            if indeg[e] == 0:
                q.append(e)  # heappush(q, -e)

    return res, idx, nr == n

def solve():
    L = list(map(int, sys.stdin.readline().split()))
    d = defaultdict(list)
    for i in range(L[1]):
        L1 = list(map(int, sys.stdin.readline().split()))
        d[L1[0]].append(L1[1])
    n = L[0]+1
    dp = [0]*n
    parent = [-1]*n

    res,idx,k = kahn(d,L[0]+1)
    # print(res)
    # print(idx)
    # print(k)

    for i in res:
        if dp[i]==0 and i!=1:
            continue
        for j in d[i]:
            if dp[i]+1>dp[j]:
                dp[j] = dp[i]+1
                parent[j] = i

    if parent[-1]==-1:
        print("IMPOSSIBLE")
        return
    
    ans = []
    i = L[0]
    while parent[i]!=-1:
        ans.append(i)
        i = parent[i]
    ans.append(1)
    ans = ans[::-1]
    print(len(ans))
    print(*ans)
    #st = sys.stdin.readline().strip()
solve()