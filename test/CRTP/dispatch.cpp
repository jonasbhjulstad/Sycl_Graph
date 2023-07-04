#include <iostream>

template <class T>
struct MyHelper
{
  size_t size() const { return sizeof(T); }

  void print(std::ostream &out) const {
    // Cast to derived, call do_print. This will be derived's do_print
    // if it exists and MyHelper's otherwise.
    static_cast<T const*>(this)->do_print(out);
  }

  void do_print(std::ostream& out) const {
    out << "(" << sizeof(T) << ")";
  }
};


struct A : MyHelper<A> {
  // no do_print, so MyHelper<A>'s will be used
  int x;
};

struct B : MyHelper<B> {
  // B's do_print is found and used in MyHelper<B>::print
  void do_print(std::ostream &out) const {
    out << "Hello!\n";
  }
};

template<typename T>
void foo(MyHelper<T> const &r) {
  r.print(std::cout);
}

int main() {
  A a;
  B b;

  foo(a);
  foo(b);
}
