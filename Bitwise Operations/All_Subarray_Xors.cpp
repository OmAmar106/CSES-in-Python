// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
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
// Segment Tree(Nodes): SegmentNode, HLD: hld
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

void fwht(vector<long long>& a, bool invert = false) {
    int n = a.size();
    for (int step = 1; step < n; step <<= 1) {
        for (int i = 0; i < n; i += (step << 1)) {
            for (int j = 0; j < step; j++) {
                long long u = a[i + j];
                long long v = a[i + j + step];
                a[i + j] = u + v;
                a[i + j + step] = u - v;
            }
        }
    }
    if (invert) {
        for (auto &x : a) x /= n;
    }
}

void fwt_or(vector<long long>& a, bool invert = false) {
    int n = a.size();
    for (int step = 1; step < n; step <<= 1) {
        for (int i = 0; i < n; i += (step << 1)) {
            for (int j = 0; j < step; j++) {
                long long u = a[i + j];
                long long v = a[i + j + step];
                if (!invert)
                    a[i + j + step] = u + v;
                else
                    a[i + j + step] = v - u;
            }
        }
    }
}

void fwt_and(vector<long long>& a, bool invert = false) {
    int n = a.size();
    for (int step = 1; step < n; step <<= 1) {
        for (int i = 0; i < n; i += (step << 1)) {
            for (int j = 0; j < step; j++) {
                long long u = a[i + j];
                long long v = a[i + j + step];
                if (!invert)
                    a[i + j] = u + v;
                else
                    a[i + j] = u - v;
            }
        }
    }
}

vector<long long> convolution(vector<long long> A, vector<long long> B) {
    int n = 1;
    int x = max(A.size(), B.size());
    while (n < x) n <<= 1;
    n <<= 1;

    A.resize(n);
    B.resize(n);

    fwht(A);
    fwht(B);

    for (int i = 0; i < n; i++)
        A[i] *= B[i];

    fwht(A, true);
    return A;
}

void solve() {
    int n;
    cin>>n;
    vi L = LII(n);

    vi pref = {0};
    for(auto it:L){
        pref.pb(pref.back()^it);
    }
    // print(pref);
    // vi pref1(pref);
    vi pref1(2e6+1,0);
    for(auto it:pref){
        pref1[it]++;
    }
    
    vi ans1 = convolution(pref1,pref1);
    vi ans;

    ans1[0] -= n+1;
    
    for(int i=0;i<2e6+1;i++){
        if(ans1[i]){
            ans.pb(i);
        }
    }

    sort(ans.begin(),ans.end());
    print(len(ans));
    for(auto it:ans){
        cout<<it<<" ";
    }
}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}