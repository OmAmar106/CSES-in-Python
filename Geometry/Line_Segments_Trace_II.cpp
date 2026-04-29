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
// Segment Tree(Nodes): SegmentNode, HLD: hld, fwht: fwht
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

// class Line1 {
// public:
//     int m, b;
//     Line1(int m = 0, int b = INF) : m(m), b(b) {}
//     int operator()(int x) const {
//         return m * x + b;
//     }
// };
// class ConvexHull {
// public:
//     int n;
//     vector<Line1> seg;
//     vector<int> lo, hi;
//     ConvexHull(int n = 1000000) : n(n), seg(4 * n, Line1()), lo(4 * n), hi(4 * n) {
//         build(1, 1, n);
//     }
//     void build(int i, int l, int r) {
//         stack<tuple<int, int, int>> stack;
//         stack.emplace(i, l, r);
//         while (!stack.empty()) {
//             auto [idx, left, right] = stack.top();
//             stack.pop();
//             lo[idx] = left;
//             hi[idx] = right;
//             seg[idx] = Line1(0, INF);
//             if (left == right) continue;
//             int mid = (left + right) / 2;
//             stack.emplace(2 * idx + 1, mid + 1, right);
//             stack.emplace(2 * idx, left, mid);
//         }
//     }
//     void insert(Line1 L) {
//         int pos = 1;
//         while (true) {
//             int l = lo[pos], r = hi[pos];
//             if (l == r) {
//                 if (L(l) < seg[pos](l)) seg[pos] = L;
//                 break;
//             }
//             int m = (l + r) / 2;
//             if (seg[pos].m < L.m) swap(seg[pos], L);
//             if (seg[pos](m) > L(m)) {
//                 swap(seg[pos], L);
//                 pos = 2 * pos;
//             } else {
//                 pos = 2 * pos + 1;
//             }
//         }
//     }
//     int query(int x) {
//         int i = 1, res = seg[i](x), pos = i;
//         while (true) {
//             int l = lo[pos], r = hi[pos];
//             res = min(res, seg[pos](x));
//             if (l == r) return res;
//             int m = (l + r) / 2;
//             pos = (x < m) ? 2 * pos : 2 * pos + 1;
//         }
//     }
// };


// struct Line {
//     int m, b, c;
//     int operator()(int x){
//         return m * x + b;
//     }
// };
// struct CHT {
//     Line dq[200000];
//     int fptr, bptr;
//     void clear(){
//         dq[0] = {0, 0, 0};
//         fptr = 0; bptr = 1;
//     }
//     bool pop_back(Line& L, Line& L1, Line& L2){
//         int v1 = (L.b - L2.b) * (L2.m - L1.m);
//         int v2 = (L2.m - L.m) * (L1.b - L2.b);
//         return (v1 == v2 ? L.c > L1.c : v1 < v2);
//     }
//     bool pop_front(Line& L1, Line& L2, int x){
//         int v1 = L1(x);
//         int v2 = L2(x);
//         return (v1 == v2 ? L1.c < L2.c : v1 > v2);
//     }
//     void insert(Line L){
//         while(bptr-fptr >= 2 && pop_back(L, dq[bptr-1], dq[bptr-2]))	bptr--;
//         dq[bptr++] = L;
//     }
//     pi query(int x){
//         while(bptr-fptr >= 2 && pop_front(dq[fptr], dq[fptr+1], x))	 fptr++;
//         return {dq[fptr](x), dq[fptr].c};
//     }
// };

struct LiChao {
    struct Node {
        int lo, hi, mid;
        long long m, b;
        bool has_line;
        Node *left, *right;
        Node(int l, int r)
            : lo(l), hi(r), mid((l + r) >> 1),
              m(0), b(0), has_line(false),
              left(nullptr), right(nullptr) {}
    };
    Node* root;
    LiChao(int l, int r) {
        root = new Node(l, r);
    }
    void add_line(long long m, long long b) {
        Node* node = root;
        while (true) {
            int lo = node->lo, hi = node->hi, mid = node->mid;
            if (!node->has_line) {
                node->m = m;
                node->b = b;
                node->has_line = true;
                return;
            }
            long long cur_m = node->m, cur_b = node->b;
            if (m * mid + b > cur_m * mid + cur_b) {
                swap(node->m, m);
                swap(node->b, b);
                cur_m = node->m;
                cur_b = node->b;
            }
            if (lo == hi) return;
            if (m * lo + b > cur_m * lo + cur_b) {
                if (!node->left)
                    node->left = new Node(lo, mid);
                node = node->left;
            }
            else if (m * hi + b > cur_m * hi + cur_b) {
                if (!node->right)
                    node->right = new Node(mid + 1, hi);
                node = node->right;
            }
            else return;
        }
    }

    long long query(int x) {
        Node* node = root;
        long long res = -1;
        while (node) {
            if (node->has_line) {
                res = max(res, node->m * x + node->b);
            }
            if (node->lo == node->hi) break;
            if (x <= node->mid) node = node->left;
            else node = node->right;
        }
        return res;
    }
};

struct SegmentTree {
    int n;
    vector<LiChao*> tree;
    SegmentTree(int n) : n(n) {
        tree.resize(4 * n);
        for (int i = 0; i < 4 * n; i++) {
            tree[i] = new LiChao(0, n);
        }
    }
    void update(int idx, int l, int r, int ql, int qr, long long m, long long c) {
        stack<tuple<int,int,int>> st;
        st.push({idx, l, r});
        while (!st.empty()) {
            auto [i, L, R] = st.top(); st.pop();
            if (R < ql || L > qr) continue;
            if (ql <= L && R <= qr) {
                tree[i]->add_line(m, c);
                continue;
            }
            int mid = (L + R) >> 1;
            st.push({2*i+1, mid+1, R});
            st.push({2*i, L, mid});
        }
    }
    long long query(int idx, int l, int r, int pos) {
        long long ans = -1;
        while (l < r) {
            int mid = (l + r) >> 1;
            ans = max(ans, tree[idx]->query(pos));
            if (pos > mid) {
                l = mid + 1;
                idx = 2*idx + 1;
            } else {
                r = mid;
                idx = 2*idx;
            }
        }
        ans = max(ans, tree[idx]->query(pos));
        return ans;
    }
};

void solve() {
    int n,m;
    cin>>n>>m;

    SegmentTree seg = SegmentTree(m+1);

    while(n--){
        int x1,y1,x2,y2;
        cin>>x1>>y1>>x2>>y2;
        int m1 = (y2-y1)/(x2-x1);
        seg.update(1,0,m,x1,x2,m1,y2-m1*x2);
        // print(1);
    }
    // print(2);
    for(int i=0;i<=m;i++){
        cout<<seg.query(1,0,m,i)<<" ";
    }
    cout<<endl;
    // print(1);
}

int32_t main() {
    fastio
    int t=1;
    while (t--) solve();
    return 0;
}