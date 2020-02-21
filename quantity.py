import numpy
import sympy

from sympy.utilities.lambdify import lambdify
from weakref import WeakValueDictionary

##############################################################
#                        Quantities                          #
##############################################################

class Quantity:
    """ Hybrid numeric and symbolic quantities. """

    # Weak registry of named quantities
    _named = WeakValueDictionary()

    # Operation lambdas
    _add = lambda a, b: a + b
    _sub = lambda a, b: a - b
    _mul = lambda a, b: a * b
    _div = lambda a, b: a / b
    _pow = lambda a, b: a**b
    _neg = lambda a: -a

    def __init__(self, data, sigma=None, sym=None, expr=None, deps=None):

        # Set data, symbol, expression
        self._data = data
        self._locked = False
        self._sym = None
        self.set_sym(sym)
        self._expr = expr

        # Set uncertainty
        if sigma is None:
            self._sigma = None
        elif isinstance(sigma, Quantity):
            self._sigma = Sigma(self, sigma.data, sigma.expr, sigma.deps)
        elif not isinstance(sigma, Sigma):
            self._sigma = Sigma(self, sigma)

        # Direct dependencies
        if deps is None:
            self.deps = set()
        else:
            self.deps = deps

    def __getattr__(self, name):

        return getattr(self._data, name)

    def __str__(self):

        return str(self._data)

    #############################
    #   Operation Definitions   #
    #############################

    def __abs__(self):

        return execute_op(abs, self)

    def __add__(self, other):

        return execute_op(Quantity._add, self, other)

    def __radd__(self, other):

        return execute_op(Quantity._add, other, self)

    def __sub__(self, other):

        return execute_op(Quantity._sub, self, other)

    def __rsub__(self, other):

        return execute_op(Quantity._sub, other, self)

    def __mul__(self, other):

        return execute_op(Quantity._mul, self, other)

    def __rmul__(self, other):

        return execute_op(Quantity._mul, other, self)

    def __truediv__(self, other):

        return execute_op(Quantity._div, self, other)

    def __rtruediv__(self, other):

        return execute_op(Quantity._div, other, self)

    def __pow__(self, other):

        return execute_op(Quantity._pow, self, other)

    def __rpow__(self, other):

        return execute_op(Quantity._pow, other, self)

    def __neg__(self):

        return execute_op(Quantity._neg, self)

    ###########################
    #   Misc. Functionality   #
    ###########################

    def _propagate_uncertainty(self):

        dep_symbols = tuple(self.expr.free_symbols)
        named_deps = tuple(Quantity._named[sym] for sym in dep_symbols)
        uncertain_deps = tuple(dep for dep in named_deps if dep.sigma is not None)
        derivs = (sympy.diff(self.expr, dep.sym) for dep in uncertain_deps)
        terms = ((deriv * dep.sigma.sym)**2 for deriv, dep in zip(derivs, uncertain_deps))
        sigma_expr = sympy.sqrt(sum(terms))

        dep_sigmas = tuple(dep.sigma for dep in uncertain_deps)
        dep_sigma_syms = tuple(sigma.sym for sigma in dep_sigmas)
        sigma_func = lambdify(dep_symbols + dep_sigma_syms, sigma_expr, 'numpy')

        sigma_deps = named_deps + dep_sigmas
        sigma_deps_data = tuple(dep.data for dep in sigma_deps)
        sigma_data = sigma_func(*sigma_deps_data)

        return Sigma(self, sigma_data, sigma_expr, sigma_deps)

    def lock_sym(self):
        """ Lock symbol to prevent name changes. """

        self._locked = True

    def set_sym(self, sym):
        """
        Set symbol for this Quantity. The Quantity be unlocked and the symbol
        must be unregistered.
        """

        if sym is None: return
        assert not self._locked, "Can only rename symbols that have not been used in any operations"

        old_sym = self._sym
        if not isinstance(sym, sympy.Symbol):
            self._sym = sympy.Symbol(sym)
        else:
            self._sym = sym

        if old_sym in Quantity._named:
            del Quantity._named[self._sym]
        assert self._sym not in Quantity._named, f"{self._sym} is already registered to a Quantity"
        Quantity._named[self._sym] = self

    ############################
    #   Read-only Properties   #
    ############################

    @property
    def data(self):

        return self._data

    @property
    def sigma(self):

        if self._sigma is None and self._expr is not None:
            self._sigma =  self._propagate_uncertainty()
        return self._sigma

    @property
    def sym(self):

        return self._sym

    @property
    def expr(self):

        if self._expr is None:
            return self.sym
        return self._expr

    @property
    def locked(self):

        return self._locked

class Sigma(Quantity):
    """ Auxiliary class for storing uncertainties. """

    def __init__(self, parent, data, expr=None, deps=()):

        if parent.sym is None:
            sym = None
        else:
            sym = f'sigma_{parent.sym}'
        super().__init__(data, sym=sym, expr=expr, deps=deps)
        self._parent = parent

    @property
    def parent(self):

        return self._parent

##############################################################
#                     Transformations                        #
##############################################################

def isquantity(arg):
    """ Check if object is an instance of Quantity. """

    return isinstance(arg, Quantity)

def numfilter(args):
    """ Convert any quantities to the underlying data. """

    return (arg.data if isquantity(arg) else arg
            for arg in args)

def symfilter(args):
    """ Convert any quantities to the underlying expression. """

    for arg in args:
        if isquantity(arg):
            yield arg.sym if arg.sym is not None else arg.expr
        else:
            yield arg

def make_deps(args):
    """ Make dependency set, and lock their symbolic representations. """

    quantities = filter(isquantity, args)
    deps = set(quantities)
    for dep in deps:
        dep.lock_sym()
    return deps

def createf(numname, symname=None):
    """
    Create a function returning a Quantity (takes numpy and sympy names). Operates on both numeric
    and symbolic components. Can opt to only supply one name if they are the same.
    """
    if symname is None: symname = numname
    numf = getattr(numpy, numname)
    symf = getattr(sympy, symname)

    def f(*args):

        data = numf(*numfilter(args))
        expr = symf(*symfilter(args))
        deps = make_deps(args)
        return Quantity(data, expr=expr, deps=deps)

    return f

def execute_op(op, *args):
    """
    Execute an operation returning a Quantity. Operates on both numeric
    and symbolic components.
    """

    data = op(*numfilter(args))
    expr = op(*symfilter(args))
    deps = make_deps(args)
    return Quantity(data, expr=expr, deps=deps)

############################
#   Function Definitions   #
############################
sign = createf('sign')
power = createf('power')
sqrt = createf('sqrt')
cbrt = createf('cbrt')
sin = createf('sin')
cos = createf('cos')
tan = createf('tan')
asin = createf('arcsin', 'asin')
acos = createf('arccos', 'acos')
atan = createf('arctan', 'atan')
exp = createf('exp')
log = createf('log')
sinh = createf('sinh')
cosh = createf('cosh')
tanh = createf('tanh')
floor = createf('floor')
ceil = createf('ceil', 'ceiling')
gcd = createf('gcd')
lcm = createf('lcm')