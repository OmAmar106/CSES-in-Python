// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
// #include <ext/pb_ds/assoc_container.hpp>
// #include <ext/pb_ds/tree_policy.hpp>
// using namespace __gnu_pbds;
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

// template <typename T>
// using ordered_multiset = tree<
//     std::pair<T, int>,
//     null_type,
//     std::less<std::pair<T, int>>,
//     rb_tree_tag,
//     tree_order_statistics_node_update
// >;
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

struct LiChao {
    int lo, hi, mid;
    int m, b;
    bool has;
    LiChao *left, *right;

    LiChao(int lo, int hi) : lo(lo), hi(hi), mid((lo + hi) >> 1),m(0), b(0), has(false),left(nullptr), right(nullptr) {}

    int f(int m, int b, int x) {
        return m * x + b;
    }

    void add_line(int nm, int nb) {
        LiChao* node = this;
        while (true) {
            if (!node->has) {
                node->m = nm;
                node->b = nb;
                node->has = true;
                return;
            }

            int lo = node->lo, hi = node->hi, mid = node->mid;
            int cm = node->m, cb = node->b;

            if (f(nm, nb, mid) > f(cm, cb, mid)) {
                swap(node->m, nm);
                swap(node->b, nb);
                cm = node->m;
                cb = node->b;
            }

            if (lo == hi) return;

            if (f(nm, nb, lo) > f(cm, cb, lo)) {
                if (!node->left)
                    node->left = new LiChao(lo, mid);
                node = node->left;
            } else if (f(nm, nb, hi) > f(cm, cb, hi)) {
                if (!node->right)
                    node->right = new LiChao(mid + 1, hi);
                node = node->right;
            } else {
                return;
            }
        }
    }

    int query(int x) {
        LiChao* node = this;
        int res = LLONG_MIN;

        while (node) {
            if (node->has)
                res = max(res, f(node->m, node->b, x));

            if (node->lo == node->hi) break;

            if (x <= node->mid)
                node = node->left;
            else
                node = node->right;
        }
        return res;
    }
};

const int SIZE = 1 << 17;
LiChao* tree[2 * SIZE];

void range_update(int left, int right, pair<int,int> val) {
    int l = left + SIZE;
    int r = right + SIZE;

    while (l <= r) {
        if (l & 1) {
            if (!tree[l])
                tree[l] = new LiChao(0, 1 << 17);
            tree[l]->add_line(val.first, val.second);
            l++;
        }
        if (!(r & 1)) {
            if (!tree[r])
                tree[r] = new LiChao(0, 1 << 17);
            tree[r]->add_line(val.first, val.second);
            r--;
        }
        l >>= 1;
        r >>= 1;
    }
}

int range_query(int pos) {
    int x = pos + 1;
    pos += SIZE;

    int ans = tree[pos] ? tree[pos]->query(x) : LLONG_MIN;

    while (pos > 1) {
        pos >>= 1;
        if (tree[pos])
            ans = max(ans, tree[pos]->query(x));
    }
    return ans;
}

void solve() {
    int q;
    cin>>q;
    while(q--){
        int ty;
        cin>>ty;
        if (ty==1) {
            int a, b;
            int l, r;
            cin>>a>>b>>l>>r;
            range_update(l-1, r-1,{a, b});
        } else {
            int x;
            cin>>x;
            int ans = range_query(x-1);
            if (ans==LLONG_MIN)cout<<"NO"<<endl;
            else cout<<ans<<endl;
        }
    }
}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}