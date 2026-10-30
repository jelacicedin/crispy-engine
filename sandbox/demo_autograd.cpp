// ============================================================================
//  60-second demo: reverse-mode autodiff for scalars, in ~150 lines of C++20.
//  NOT the start of crispy-engine -- this is a sandbox toy. Poke it, break it,
//  delete it. Five ops (+ - * / exp) and tanh falls out for free.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

struct Value;
using V = std::shared_ptr<Value>;  // Rule of Zero: no dtor, no copy ctor, no move ctor

struct Value {
  double data = 0.0;
  double grad = 0.0;
  std::string op;                             // "" == leaf
  std::vector<std::pair<V, double>> parents;  // (input, d(this)/d(input))
  explicit Value(double d) : data{d} {}
};

static V make(double d) { return std::make_shared<Value>(d); }
static V make(const char *label, double d) {
  auto v = make(d);
  v->op = label;
  return v;
}

// ---- the complete op set ---------------------------------------------------
static V binary(const char *name, const V &a, const V &b, double d, double ga, double gb) {
  auto o = make(d);
  o->op = name;
  o->parents = {{a, ga}, {b, gb}};
  return o;
}
static V operator+(const V &a, const V &b) { return binary("+", a, b, a->data + b->data, 1.0, 1.0); }
static V operator-(const V &a, const V &b) { return binary("-", a, b, a->data - b->data, 1.0, -1.0); }
static V operator*(const V &a, const V &b) { return binary("*", a, b, a->data * b->data, b->data, a->data); }
static V operator/(const V &a, const V &b) {
  double d = a->data / b->data;
  return binary("/", a, b, d, 1.0 / b->data, -d / b->data);
}
static V exp(const V &a) {
  double d = std::exp(a->data);
  auto o = make(d);
  o->op = "exp";
  o->parents = {{a, d}};
  return o;
}
// scalar sugar -- only the overloads that get used, so nothing is dead code
static V operator+(double a, const V &b) { return make(a) + b; }
static V operator-(const V &a, double b) { return a - make(b); }
static V operator-(double a, const V &b) { return make(a) - b; }
static V operator*(const V &a, double b) { return a * make(b); }
static V operator*(double a, const V &b) { return make(a) * b; }
static V operator/(const V &a, double b) { return a / make(b); }

static V abs_(const V &a) {
  auto o = make(std::abs(a->data));
  o->op = "abs";
  o->parents = {{a, a->data >= 0.0 ? 1.0 : -1.0}};
  return o;
}
static V sign_(const V &a) {
  auto o = make(a->data > 0.0 ? 1.0 : (a->data < 0.0 ? -1.0 : 0.0));
  o->op = "sign";
  o->parents = {{a, 0.0}};
  return o;
}

// tanh is NOT special-cased. It's just exp, because the chain rule composes.
// Written as sign(x)*(1-e)/(1+e) with e = exp(-2|x|), so it saturates instead of
// overflowing: the naive (e^(2x)-1)/(e^{2x}+1) returns inf/inf for |x| > ~355.
static V tanh_(const V &x) {
  auto e = exp(-2.0 * abs_(x));
  return sign_(x) * ((1.0 - e) / (1.0 + e));
}

// ---- the whole backward pass ----------------------------------------------
static void backward(const V &root) {
  std::vector<V> topo;
  std::unordered_set<const Value *> seen;
  std::function<void(const V &)> build = [&](const V &v) {
    if (!seen.insert(v.get()).second) return;
    for (auto &p : v->parents) build(p.first);
    topo.push_back(v);
  };
  build(root);

  for (auto &v : topo) v->grad = 0.0;  // zero all grads in one shot
  root->grad = 1.0;                    // d(loss)/d(loss) == 1
  for (auto it = topo.rbegin(); it != topo.rend(); ++it)
    for (auto &[in, local] : (*it)->parents) in->grad += (*it)->grad * local;
}

// ---- pretty printing -------------------------------------------------------
static void print_graph(const V &root) {
  std::unordered_set<const Value *> seen;
  std::function<void(const V &, std::string)> walk = [&](const V &v, std::string pad) {
    const std::string name = v->op.empty() ? "leaf" : v->op;
    if (!seen.insert(v.get()).second) {
      std::printf("%s%-5s \033[2m(shared -- gradient accumulates here)\033[0m\n", pad.c_str(), name.c_str());
      return;
    }
    std::printf("%s%-5s data=% .4f\n", pad.c_str(), name.c_str(), v->data);
    for (auto &p : v->parents) walk(p.first, pad + "   ");
  };
  walk(root, "");
}

static std::string sparkline(const std::vector<double> &ys) {
  const char *blocks[] = {"\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588"};
  double lo = 1e9, hi = -1e9;
  std::vector<double> norm;
  for (double y : ys) {  // log scale: losses live across many orders of magnitude
    double l = std::log10(y + 1e-12);
    norm.push_back(l);
    lo = std::min(lo, l);
    hi = std::max(hi, l);
  }
  std::string out;
  for (double l : norm) {
    int idx = (hi - lo < 1e-12) ? 0 : int(7.0 * (l - lo) / (hi - lo));
    out += blocks[idx];
  }
  return out;
}

// ============================================================================
int main() {
  std::printf("\n\033[1mReverse-mode autodiff, from scratch\033[0m -- 5 ops: + - * / exp\n\n");

  // ---- 1. the graph is a DAG, not a tree ---------------------------------
  std::printf("\033[1m1. the graph\033[0m   f(x, y) = x*y + x*1.0\n\n");
  auto x = make("x", 3.0), y = make("y", 2.0);
  auto f = x * y + x * 1.0;
  print_graph(f);
  backward(f);
  std::printf("\n   df/dx = %.4f (= y + 1 = 3)\n   df/dy = %.4f (= x = 3)\n\n", x->grad, y->grad);

  // ---- 2. finite-difference gradient check ------------------------------
  std::printf("\033[1m2. gradient check\033[0m  against central differences\n\n");
  auto w = make("w", 0.7), b = make("b", 0.1), x1 = make("x1", 0.5);
  auto loss = [&] { return tanh_(w * x1 + b) * tanh_(w * x1 + b); };
  backward(loss());
  const double ana_w = w->grad, ana_b = b->grad;
  const double h = 1e-6;
  auto numeric = [&](V &p) {
    p->data += h;  double up = loss()->data;
    p->data -= 2 * h; double dn = loss()->data;
    p->data += h;
    return (up - dn) / (2 * h);
  };
  std::printf("   %-4s analytic=% .10f  numeric=% .10f  err=%.2e\n", "w", ana_w, numeric(w), std::abs(ana_w - numeric(w)));
  std::printf("   %-4s analytic=% .10f  numeric=% .10f  err=%.2e\n\n", "b", ana_b, numeric(b), std::abs(ana_b - numeric(b)));

  // ---- 3. train an MLP on XOR -------------------------------------------
  std::printf("\033[1m3. train a 2-4-1 MLP on XOR\033[0m  (tanh, MSE, plain SGD)\n\n");
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  std::vector<std::vector<V>> W1(4, std::vector<V>(2));
  std::vector<V> b1(4), W2(4);
  for (auto &row : W1) for (auto &wv : row) wv = make(U(rng));
  for (auto &bv : b1) bv = make(U(rng));
  for (auto &wv : W2) wv = make(U(rng));
  V b2 = make(U(rng));

  const std::vector<std::pair<double, double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  const std::vector<double> Y = {1.0, -1.0, -1.0, 1.0};  // tanh targets

  auto forward = [&](double a, double bb) {
    V xa = make(a), xb = make(bb);
    std::vector<V> hidden;
    for (int j = 0; j < 4; ++j) hidden.push_back(tanh_(W1[j][0] * xa + W1[j][1] * xb + b1[j]));
    V out = b2;
    for (int j = 0; j < 4; ++j) out = out + hidden[j] * W2[j];
    return out;
  };

  std::vector<double> history;
  const int EPOCHS = 4000;
  const double lr = 0.10;
  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    V total = make(0.0);
    for (size_t i = 0; i < X.size(); ++i) {
      auto d = forward(X[i].first, X[i].second) - Y[i];
      total = total + d * d;
    }
    history.push_back(total->data / X.size());
    backward(total / double(X.size()));  // d(mean loss), not d(sum loss)
    auto step = [&](V &p) { p->data -= lr * p->grad; };
    for (auto &row : W1) for (auto &wv : row) step(wv);
    for (auto &bv : b1) step(bv);
    for (auto &wv : W2) step(wv);
    step(b2);
  }

  std::printf("   loss %zu epochs: %.2e -> %.2e\n\n", history.size(), history.front(), history.back());
  std::vector<double> curve;
  for (size_t i = 0; i < history.size(); i += history.size() / 60) curve.push_back(history[i]);
  std::printf("   %s\n\n", sparkline(curve).c_str());
  for (size_t i = 0; i < X.size(); ++i) {
    double got = forward(X[i].first, X[i].second)->data;
    std::printf("   %0.f XOR %0.f  ->  % .4f   (target % .1f)\n", X[i].first, X[i].second, got, Y[i]);
  }
  std::printf("\n\033[1mdone.\033[0m no autograd library, no framework, no graph compiler.\n\n");
  return 0;
}
