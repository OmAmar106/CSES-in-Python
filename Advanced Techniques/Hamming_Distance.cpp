// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
using namespace std;
#define INF LLONG_MAX
#define fastio ios::sync_with_stdio(false); cin.tie(0);
#define print(arr) for (auto it : arr){cout<<it<<" ";}cout<<endl;
#define len(arr) arr.size()
#define printf(x) cout << x << endl;
#define printm(map) cout<<"{";for(auto it: map){cout<<it.first<<":"<<it.second<<",";};cout<<"}"<<endl;

// const int MOD = 998244353;
const int MOD = 1e9 + 7;

int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}

int lcm(int a, int b) {
    return a / gcd(a, b) * b;
}

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable
// Segment Tree(lazy propogation): SegmentTree
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, Trie/Treap: Tries
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, DSU: DSU, Geometry: Geometry, FFT: fft
// Persistent Segment Tree: perseg, FreqGraphs: bgraph
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

int func(string &s){
    int k =0;
    for(int i=len(s)-1;i>=0;i--){
        k *= 2;
        k += (s[i]-'0');
    }
    return k;
}

void solve() {
    int n,k; cin >> n>>k;
    vector<int> L;
    int ans = 32;
    for(int i=0;i<n;i++){
        string s1;
        cin>>s1;
        L.push_back(func(s1));
        for(int j=0;j<i;j++){
            ans = min(ans,(int)__builtin_popcount(L[i]^L[j]));
        }
        if(ans==0){
            break;
        }
    }
    printf(ans)
}

int32_t main() {
    fastio
    int t=1;
    while (t--) solve();
    return 0;
}