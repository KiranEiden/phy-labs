""" Speedier version of unit expression parser for UnitContainer class. """

import re
from quantity import Unit

class UnitContainer:

    # Unit string (e.g. 'cm')
    unit_regex = re.compile(r"(?=\w)[^\d_]\w*")
    # Exponentiation operator (e.g. '**' in 'cm**2')
    expop_regex = re.compile(r"\*\*|\^")
    # Multiplication and division
    op_regex = re.compile(r"(?!\*\*)(?<!\*)[*/]")
    # The exponent itself (a number)
    exponent_regex = re.compile(r"-?[\d.]+")

    def __init__(self, unitstr):

        units = list(self.unit_regex.finditer(unitstr))
        expops = list(self.expop_regex.finditer(unitstr))
        ops = list(self.op_regex.finditer(unitstr))
        exps = list(self.exponent_regex.finditer(unitstr))
        matches = sorted(units + expops + ops + exps, key=lambda m: m.start())

        try:
            self.units = dict()
            for unit, exponent in self.extract_units(matches):
                self.units.setdefault(unit, 0)
                self.units[unit] += exponent
        except AssertionError:
            raise ValueError("Error parsing unit string")
        except ValueError as e:
            raise ValueError("Invalid unit power: " + str(e))
        except KeyError as e:
            raise ValueError("Unknown unit: " + str(e))

    def extract_units(self, matches):
        """ Extract units and exponents from a unit definition string. """

        assert matches[0].re == self.unit_regex

        unit = None
        neg = False
        last = None

        for m in matches:

            string = m.group()

            if last == self.unit_regex:
                assert (m.re == self.op_regex) or (m.re == self.expop_regex)
            elif last == self.expop_regex:
                assert m.re == self.exponent_regex
            elif last == self.op_regex:
                assert m.re == self.unit_regex
            elif last == self.exponent_regex:
                assert m.re == self.op_regex

            if m.re == self.unit_regex:

                unit = Unit.get(string)

            elif m.re == self.op_regex:

                if last == self.unit_regex:
                    yield unit, -1.0 if neg else 1.0
                    unit = None
                    neg = False

                if string == '/':
                    neg = True

            elif m.re == self.exponent_regex:

                exponent = float(string)
                if neg: exponent = -exponent
                yield unit, exponent

                unit = None
                neg = False

            last = m.re

        if last == self.unit_regex:
            yield unit, -1.0 if neg else 1.0
        else:
            assert last == self.exponent_regex