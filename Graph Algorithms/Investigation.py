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

def solve():
    L = list(map(int, sys.stdin.readline().split()))
    d = defaultdict(list)
    for i in range(L[1]):
        L1 = list(map(int, sys.stdin.readline().split()))
        d[L1[0]].append((L1[1],L1[2]))

    dist = [[float('inf'),0,float('inf'),0] for i in range(L[0]+1)]
    dist[1][0] = 0
    dist[1][1] = 1
    dist[1][2] = 0
    dist[1][3] = 0
    queue = [(0, 1)]

    while queue:
        path_len, v = heappop(queue)
        if path_len==dist[v][0]:
            for w, edge_len in d[v]:
                if edge_len + path_len < dist[w][0]:
                    dist[w][0] = edge_len + path_len
                    dist[w][1] = dist[v][1]
                    dist[w][2] = dist[v][2]+1
                    dist[w][3] = dist[v][3]+1
                    heappush(queue, (edge_len + path_len, w))
                elif edge_len+path_len==dist[w][0]:
                    dist[w][1] += dist[v][1]
                    dist[w][1] %= MOD
                    dist[w][2] = min(dist[w][2],dist[v][2]+1)
                    dist[w][3] = max(dist[w][3],dist[v][3]+1)
    
    print(*dist[-1])
    #st = sys.stdin.readline().strip()
solve()