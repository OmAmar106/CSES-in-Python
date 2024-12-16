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
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree,Other, bootstrap : bootstrap
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
    n = int(sys.stdin.readline().strip())
    d = defaultdict(list)
    for i in range(n-1):
        L = list(map(int, sys.stdin.readline().split()))
        d[L[0]].append(L[1])
        d[L[1]].append(L[0])
    dp = defaultdict(list)

    def rec(cur,prev):
        stack = [(1, -1, 0)]
        return_values = {}
        while stack:
            cur, prev, phase = stack.pop()
            if phase == 0:
                dp[cur].append((0, -1))
                stack.append((cur, prev, 1))
                for child in d[cur]:
                    if child != prev:
                        stack.append((child, cur, 0))
            elif phase == 1:
                for child in d[cur]:
                    if child != prev:
                        dp[cur].append((return_values[(child, cur)], child))
                dp[cur].sort()
                ans[cur - 1] = dp[cur][-1][0]
                return_values[(cur, prev)] = 1 + dp[cur][-1][0]

    ans = [0 for i in range(n)]
    rec(1,0)

    def rec1(cur,prev,fromabove):
        stack = [(cur, 0, fromabove, 0)]
        visited_stack = []
        while stack:
            cur, prev, fromabove, phase = stack.pop()
            if phase == 0:
                visited_stack.append((cur, prev, fromabove))
                for i in d[cur]:
                    if i == prev:
                        continue
                    if i != dp[cur][-1][1]:
                        k2 = dp[cur][-1][0]
                    else:
                        k2 = dp[cur][-2][0]
                    stack.append((i, cur, max(fromabove + 1, k2 + 1), 0))
                stack.append((cur, prev, fromabove, 1))
            elif phase == 1:
                cur, prev, fromabove = visited_stack.pop()
                ans[cur - 1] = max(ans[cur - 1], fromabove)
        return ans
    
    rec1(1,-1,0)
    print(*ans)
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()