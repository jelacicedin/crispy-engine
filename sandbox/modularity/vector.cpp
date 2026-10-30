export module Vector;

export class Vector {
public:
    Vector(int s);
    double &operator[](int i);
    const double &operator[](int i) const;

    Vector(const Vector &a);            // copy constructor
    Vector &operator=(const Vector &a); // copy assignment

    Vector(Vector &&a);            // move constructor
    Vector &operator=(Vector &&a); // move assignment

    int size() const;
    ~Vector() { delete[] elem; }

private:
    double *elem;
    int sz;
};

Vector::Vector(int s) : elem{new double[s]}, sz{s} {}
double &Vector::operator[](int i) { return elem[i]; }
const double &Vector::operator[](int i) const { return elem[i]; }
int Vector::size() const { return sz; }

// Copy constructor - bad for large containers
Vector::Vector(const Vector &a) : elem{new double[a.sz]}, sz{a.sz} {
    for (int i = 0; i < a.sz; i++) {
        elem[i] = a.elem[i];
    }
}

// Copy assignment - bad for large containers
Vector &Vector::operator=(const Vector &a) {
    double *temp = new double[a.sz];
    for (int i = 0; i < a.sz; i++)
        temp[i] = a.elem[i];
    delete[] elem; // wipe old elements
    elem = temp;
    sz = a.sz;
    return *this;
}

// Move constructor
Vector::Vector(Vector &&a) : elem{a.elem}, sz{a.sz} {
    a.elem = nullptr;
    a.sz = 0;
}



export bool operator==(const Vector &v1, const Vector &v2) {
    if (v1.size() != v2.size())
        return false;
    for (int i = 0; i < v1.size(); i++) {
        if (v1[i] != v2[i])
            return false;
    }
    return true;
}