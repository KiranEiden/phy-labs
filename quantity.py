import numpy
import sympy
import re

from functools import reduce
from collections import namedtuple
from enum import Enum

from sympy.utilities.lambdify import lambdify
import sympy.parsing.sympy_parser as symparse

# TODO: Define Quantity.__format__
# TODO: Add unit support
# TODO: Add definitions of constants

class Ops(Enum):
    """ Operation lambdas """

    ADD = lambda a, b: a + b
    SUB = lambda a, b: a - b
    MUL = lambda a, b: a * b
    DIV = lambda a, b: a / b
    POW = lambda a, b: a ** b
    NEG = lambda a: -a

##############################################################
#                        Quantities                          #
##############################################################

class Quantity:
    """ Hybrid numeric and symbolic quantities. """

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

    def __repr__(self):

        if self.sym is None:
            symstr = ""
        else:
            symstr = f"{self.sym} = "

        if self.sigma is None:
            sigstr = ""
        else:
            sigstr = f" ± {self.sigma.data}"

        return f"{symstr}{self.data}{sigstr}"

    #############################
    #   Operation Definitions   #
    #############################

    def __abs__(self):

        return execute_op(abs, self)

    def __add__(self, other):

        return execute_op(Ops.ADD, self, other)

    def __radd__(self, other):

        return execute_op(Ops.ADD, other, self)

    def __sub__(self, other):

        return execute_op(Ops.SUB, self, other)

    def __rsub__(self, other):

        return execute_op(Ops.SUB, other, self)

    def __mul__(self, other):

        return execute_op(Ops.MUL, self, other)

    def __rmul__(self, other):

        return execute_op(Ops.MUL, other, self)

    def __truediv__(self, other):

        return execute_op(Ops.DIV, self, other)

    def __rtruediv__(self, other):

        return execute_op(Ops.DIV, other, self)

    def __pow__(self, other):

        return execute_op(Ops.POW, self, other)

    def __rpow__(self, other):

        return execute_op(Ops.POW, other, self)

    def __neg__(self):

        return execute_op(Ops.NEG, self)

    ###########################
    #   Misc. Functionality   #
    ###########################

    def _propagate_uncertainty(self):

        deps = tuple(self.deps)
        dep_symbols = tuple(dep.sym for dep in deps)
        uncertain_deps = tuple(dep for dep in deps if dep.sigma is not None)
        derivs = (sympy.diff(self.expr, dep.sym) for dep in uncertain_deps)
        terms = ((deriv * dep.sigma.sym)**2 for deriv, dep in zip(derivs, uncertain_deps))
        sigma_expr = sympy.sqrt(sum(terms))

        dep_sigmas = tuple(dep.sigma for dep in uncertain_deps)
        dep_sigma_syms = tuple(sigma.sym for sigma in dep_sigmas)
        sigma_func = lambdify(dep_symbols + dep_sigma_syms, sigma_expr, 'numpy')

        sigma_deps = deps + dep_sigmas
        sigma_deps_data = tuple(dep.data for dep in sigma_deps)
        sigma_data = sigma_func(*sigma_deps_data)

        return Sigma(self, sigma_data, sigma_expr, sigma_deps)

    def lock_sym(self):
        """ Lock symbol to prevent name changes. """

        self._locked = True

    def set_sym(self, sym):
        """ Set symbol for this Quantity. The Quantity must be unlocked. """

        if sym is None: return
        assert not self._locked, "Can only rename symbols that have not been used in any operations"

        if not isinstance(sym, sympy.Symbol):
            self._sym = sympy.Symbol(sym)
        else:
            self._sym = sym

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

#########################################################
#                        Units                          #
#########################################################

class Dimension:

    # Dimension registry
    _dreg = dict()

    def __new__(cls, desc):

        if desc in Dimension._dreg:
            return Dimension._dreg[desc]
        else:
            inst = super().__new__(cls)
            Dimension._dreg[desc] = inst
            return inst

    def __init__(self, desc):

        self._desc = desc
        self._base = None

    def __repr__(self):

        return f"Dimension('{self._desc}')"

    def __hash__(self):

        return hash(self._desc)

    def __eq__(self, other):

        return repr(self) == repr(other)

    @classmethod
    def get(cls, desc):

        return cls._dreg[desc]

    @classmethod
    def exists(cls, desc):

        return desc in cls._dreg

    def set_base_unit(self, unit):

        assert self._base is None, f"Base unit already set for {self}!"
        self._base = unit

    @property
    def description(self):

        return self._desc

    @property
    def base_unit(self):

        return self._base

UnitConversion = namedtuple("UnitConversion", "scale offset")

class Unit:

    # Unit registry
    _ureg = dict()

    def __new__(cls, full_name, defn, aliases=()):

        if not full_name:
            raise ValueError("Unit must have a name!")
        names = (full_name,) + aliases
        registered = [name in Unit._ureg for name in names]

        if any(registered):

            existing_names = list(filter(lambda name: name in Unit._ureg, names))
            existing_units = [Unit._ureg[name] for name in existing_names]
            for i in range(len(existing_units)-1):
                if existing_units[i] != existing_units[i+1]:
                    raise ValueError(f"{existing_names} exist with conflicting definitions.")

            if not registered[0]:
                raise ValueError("Unit alias registered, but full name is not. Try using " +
                                 "Unit.rename to reset the full name.")

            if all(registered):
                return Unit._ureg[full_name]

            unit = Unit._ureg[full_name]
            for name in names:
                Unit._ureg[name] = unit
            return unit

        else:

            inst = super().__new__(cls)
            for name in names:
                Unit._ureg[name] = inst
            return inst

    def __init__(self, full_name, defn, aliases=()):

        self._full_name = full_name
        self._aliases = list(aliases)

        self._from_base = None
        m = re.fullmatch('base_(?P<dim>\w+)', defn)

        if m:
            self._dim = Dimension.get(m.groupdict()['dim'])
            self._dim.set_base_unit(self)
            self._from_base = UnitConversion(scale=1.0, offset=0.0)
        else:
            self._dim, self._from_base = self.parse_defn(defn)

    def parse_defn(self, defn):
        """
        Parse definition string, returning dimensionality and UnitConversion in terms of base unit
        for that dimensionality.
        """

        transformations = symparse.standard_transformations + \
                          (symparse.convert_xor, symparse.implicit_multiplication)
        expr = symparse.parse_expr(defn, transformations=transformations)
        assert len(expr.free_symbols) == 1, "Must define unit in terms of only one other unit"

        expr = sympy.Poly(expr, domain="RR")
        conv = UnitConversion(*expr.all_coeffs())

        unitsym, = expr.free_symbols
        unitsym = str(unitsym)
        unit = Unit.get(unitsym)

        # a*some_unit + b => a*(c*base_unit + d) + b
        a, b = conv
        c, d = unit.from_base
        scale = a*c
        offset = a*d + b

        return unit.dimensionality, UnitConversion(scale, offset)

    def __repr__(self):

        return f"Unit('{self._full_name}')"

    def __hash__(self):

        return hash(self._full_name)

    def __eq__(self, other):

        return self.full_name == other.full_name

    @classmethod
    def get(cls, name):

        return cls._ureg[name]

    @classmethod
    def exists(cls, name):

        return name in cls._ureg

    @classmethod
    def rename(cls, oldname, newname):

        unit = cls.get(oldname)
        unit._full_name = newname

        for name in unit.names:
            cls._ureg[name] = unit

    @classmethod
    def remove(cls, name):

        unit = cls.get(name)

        for name in unit.names:
            del cls._ureg[name]

    @classmethod
    def add_alias(cls, name, alias):

        unit = cls.get(name)

        if alias not in unit.names:
            unit.aliases.append(alias)
            cls._ureg[alias] = unit

    @classmethod
    def remove_alias(cls, alias):

        unit = cls.get(alias)

        if unit.full_name != alias:
            unit.aliases.remove(alias)
            del cls._ureg[alias]

    @property
    def full_name(self):

        return self._full_name

    @property
    def aliases(self):

        return self._aliases

    @property
    def names(self):

        return [self._full_name] + self._aliases

    @property
    def dimensionality(self):

        return self._dim

    @property
    def from_base(self):

        return self._from_base

class UnitContainer:

    def __init__(self, unitstr):

        self.units = dict()

        for unit, exponent in self.extract_units(unitstr):
            unit = Unit.get(unit)
            self.units[unit] = exponent

    def extract_units(self, unitstr):
        """ Extract units and exponents from a unit definition string. """

        transformations = symparse.standard_transformations + (symparse.convert_xor,)
        expr = symparse.parse_expr(unitstr, transformations=transformations)

        yield from self._traverse_expr(expr)

    def _traverse_expr(self, expr):

        if isinstance(expr, sympy.Pow):

            a, b = expr.args
            if not (isinstance(a, sympy.Symbol) and isinstance(b, sympy.Number)):
                raise ValueError(f"{a}**{b} not allowed in unit expression, must be symbol to some power.")
            yield str(a), float(b)

        elif isinstance(expr, sympy.Symbol):

            yield str(expr), 1.0

        elif isinstance(expr, sympy.Mul):

            for arg in expr.args:
                yield from self._traverse_expr(arg)

        elif isinstance(expr, sympy.Number):

            raise ValueError(f"Non-exponent number {expr} in unit expression not permitted.")

        else:

            raise ValueError(f"Invalid operation {expr.func.__name__} in unit expression.")


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

def set_union(a, b):

    return a | b

def make_deps(args):
    """ Make dependency set, and lock their symbolic representations. """

    quantities = filter(isquantity, args)
    deps = (q.deps if q.sym is None else {q} for q in quantities)
    deps = reduce(set_union, deps)
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

Dimension("length")
Unit("meter", "base_length", ("m",))
Unit("centimeter", "9/5 * (32 - m)", ("cm",))