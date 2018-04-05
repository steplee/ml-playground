#include <iostream>
#include <vector>
#include <memory>
#include <utility>

#include "xtensor/xarray.hpp"
#include "util.h"

using namespace std;


enum Type {
  Float=0, Double=1, Int=2 
};
constexpr int TYPE_SIZES[] = {sizeof(float), sizeof(double), sizeof(int)};
#define Size_of(T) (TYPE_SIZES[(T)])


class Node {
  public:
    using ptr = shared_ptr<Node>;

    //void backward();
};

typedef vector<Node::ptr> NodePtrVec;

// ---------- Tensor ----------------------------

class Tensor : public Node {
  Type type;
  int size;
  vector<int> dims;
  void* buf;
  bool dynamic;

  public:

    Tensor(Type type, int size, bool dynamic=true, vector<int> dims={});

    using ptr = shared_ptr<Tensor>;
};

Tensor::Tensor(Type type_, int size_, bool dynamic_, vector<int> dims_) :
  type(type_), size(size_), dynamic(dynamic_), dims(dims_)
{
  if (dims.size() == 0) dims = {size};
  assert(prod<int>(dims) == size);
  buf = malloc(Size_of(type) * size);
  assert(buf);
}

Tensor::ptr floats(int size) {
  return make_shared<Tensor>(Float, size);
}


// ---------- OpNode ----------------------------

class OpNode : public Node {
  public:
    OpNode(NodePtrVec prevs);
    NodePtrVec prevs;
};
OpNode::OpNode(NodePtrVec prevs_) : prevs(prevs_) {}

// ---------- Softmax ----------------------------

class SoftmaxNode : public OpNode {
  public:
    SoftmaxNode(NodePtrVec prevs);
};
SoftmaxNode::SoftmaxNode(NodePtrVec prevs) : OpNode(prevs) {}

SoftmaxNode::ptr softmax(Tensor::ptr t) {
  return make_shared<SoftmaxNode>(NodePtrVec({t}));
}


// ---------- Linear Node ----------------------------

// Implicitly defines its weights
class LinearNode : public OpNode {
  public:
    LinearNode(Node::ptr a, Node::ptr b);

    Tensor::ptr operator()(Tensor::ptr in);
};
LinearNode::LinearNode(Node::ptr a, Node::ptr b) : OpNode({a,b}) {}

LinearNode::ptr linear(Node::ptr a, Node::ptr b) {
  return make_shared<LinearNode>(a,b, out_size);
}

Tensor::ptr LinearNode::operator()(Tensor::ptr in) {
  //auto out = ndtensor(
}


int main() {
  cout << "hello world!\n";

  /*
  x = floats(20);
  w0 = weights(20,10);
  w1 = weights(10,1);
  y = softmax(w1);
  loss = xent(y,y_true);

  optim(loss);
  */

  return 0;
}
