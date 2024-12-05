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
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU
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
    d1 = defaultdict(list)
    for i in range(L[1]):
        L1 = list(map(int, sys.stdin.readline().split()))
        d[L1[0]].append(L1[1])
        d1[L1[1]].append(L1[0])
    
    @bootstrap
    def rec(cur,d):
        visited[cur] = True
        ans = 0
        for i in d[cur]:
            if visited[i]:
                continue
            ans = (yield rec(i,d))
        yield ans
    
    visited = [False for i in range(L[0]+1)]
    rec(1,d)
    
    for j in range(1,L[0]+1):
        if not visited[j]:
            print("NO")
            print(1,j)
            return
    visited = [False for i in range(L[0]+1)]
    rec(1,d1)
    for j in range(1,L[0]+1):
        if not visited[j]:
            print("NO")
            print(j,1)
            return
    print("YES")
    #st = sys.stdin.readline().strip()
solve()