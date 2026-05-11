// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
#include <climits>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;

#define INF LLONG_MAX
#define fastio ios::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr);
#define len(arr) arr.size()
#define f first
#define s second
#define pb push_back
#define all(x) x.begin(), x.end()
#define range(i, n) for (int i = 0; i < (n); ++i)
#define rangea(i, a, b) for (int i = (a); i < (b); ++i)
#define int long long
#define MOD1 998244353
#define MOD 1000000007
using vi = vector<int>;
using vii = vector<vi>;
using viii = vector<vii>;
using ll = long long;
using pi = pair<int,int>;
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

#ifdef LOCAL
#define dbg(x...) cout<<"["<<#x<<"] = "; print(x);
#else
#define dbg(x...)
#endif

template <typename T>
using ordered_multiset = tree<
    std::pair<T, int>,
    null_type,
    std::less<std::pair<T, int>>,
    rb_tree_tag,
    tree_order_statistics_node_update
>;
template<class T> bool chmin(T& a,const T& b){ if(b<a){ a=b; return 1;} return 0; }
template<class T> bool chmax(T& a,const T& b){ if(a<b){ a=b; return 1;} return 0;}
template<class T> using minpq = priority_queue<T,vector<T>,greater<T>>;
template<class T> using maxpq = priority_queue<T>;
template<typename T> istream& operator>>(istream& in, vector<T>& v) {for (auto& x : v) in >> x;return in;}
template<typename T1, typename T2> istream& operator>>(istream& in, pair<T1, T2>& p) {in >> p.first >> p.second;return in;}
template<class T> void _print(const T &x){ cout << x; }
template<class T,class U> void _print(const pair<T,U> &p){cout<<"(";_print(p.first);cout<<",";_print(p.second);cout<<")";}
template<class T>
void _print(const vector<T> &v){
    cout<<"[";bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"]";
}
template<class T>
void _print(const set<T> &v){
    cout<<"{";
    bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"}";
}
template<class T>
void _print(const unordered_set<T> &v){
    cout<<"{";
    bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"}";
}
template<class T,class U>
void _print(const unordered_map<T,U> &m){
    cout<<"{";
    bool f=0;
    for(const auto &p: m){
        if(f) cout<<", ";
        _print(p);
        f=1;
    }
    cout<<"}";
}
template<class T,class U>
void _print(const map<T,U> &m){
    cout<<"{";
    bool f=0;
    for(const auto &p: m){
        if(f) cout<<", ";
        _print(p);
        f=1;
    }
    cout<<"}";
}
void print(){ cout<<"\n"; }
template<class T,class... Args>
void print(const T& a,const Args&... args){
    _print(a);
    if constexpr(sizeof...(args)) cout<<" ";
    print(args...);
}

int gcd(int a, int b) {if (b == 0) return a;return gcd(b, a % b);}
int lcm(int a, int b) {return a / gcd(a, b) * b;}
int II(){int a;cin>>a;return a;}
string SI(){string s;cin>>s;return s;}
vi LII(int n){vi a(n);cin>>a;return a;}
int rnd(int l,int r){return uniform_int_distribution<int>(l,r)(rng);}

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable, SortedList: sortl
// Segment Tree(lazy propogation): SegmentTree, Merge Sort Tree: sorttree, Trie/Treap: Tries
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, Centroid Decomp.: graph_decom
// Persistent Segment Tree: perseg, FreqGraphs: bgraph, GrapthOth: graphoth, DSU: DSU, FFT:fft
// Rollback/Par DSU: rbdsu, treap: treap, graphflow(mat_match): graphflow, Persistent Seg Tree: perseg
// Segment Tree(Nodes): SegmentNode, HLD: hld, fwht: fwht
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

void solve() {
    int n,k3;
    cin>>n>>k3;

    vi L = LII(n);

    vii cost(n+1,vi (n+1,INF));

    for(int i=0;i<len(L);i++){
        int med = i-1;
        int sumi = 0,sumi1 = 0;
        int val = 0;
        for(int j=i;j<len(L);j++){
            sumi += L[j];
            val += L[j]*(j-med);
            while(sumi1*2<sumi){
                val += sumi1;
                val -= (sumi-sumi1);
                med += 1;
                sumi1 += L[med];
            }
            cost[i+1][j+1] = val;
        }
    }

    auto DnC = [](vi &prev, vi& dp,vii& cost){
        auto solve = [](auto self,vi &prev,vi& dp,vii& cost,int l,int r,int opt_l,int opt_r){
            if(l>r){return;}
            int mid = (l+r)/2;
            int best_k = -1;
            int best_val = INF;
            int end = max(0ll,min(mid-1,opt_r));

            for(int k=opt_l;k<end+1;k++){
                if(cost[k+1][mid]==INF || prev[k]==INF){continue;}
                int val = prev[k]+cost[k+1][mid];
                if(val<best_val){
                    best_val = val;
                    best_k = k;
                }
            }
            if(best_k==-1){
                self(self,prev,dp,cost,mid+1,r,best_k,opt_r);
                return;
            }
            dp[mid] = best_val;
            self(self,prev,dp,cost,l,mid-1,opt_l,best_k);
            self(self,prev,dp,cost,mid+1,r,best_k,opt_r);
        };
        solve(solve,prev,dp,cost,0,len(dp)-1,0,len(dp)-1);
    };

    vi prev(n+1,INF);
    prev[0] = 0;
    vi dp(n+1,INF);

    while(k3--){
        DnC(prev,dp,cost);
        swap(dp,prev);
    }

    print(prev[n]);

}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}