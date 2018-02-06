import unittest
from modules.fitting import *

THRESHOLD = 1e-10

class TestFitting(unittest.TestCase):

    def fit_assertion(self, x, y, fun, res):

        fit = fixed_params(fun, *res[0])
        self.assertLess(np.mean([abs(fit(x_i) - y_i) for x_i, y_i in zip(x, y)]) / np.mean(y), THRESHOLD)
        self.assertLess(np.mean([err_p / p for err_p, p in zip(res[1], res[0])]), THRESHOLD)

    def test_lorentzian_fit(self):

        x = [16, 17.5, 18.2, 18.6, 19.2, 19.8, 20.4]
        params = (8.0, 1.0, 18.62)
        fun = fixed_params(lorentzian, *params)
        y = list(map(fun, x))
        res = lorentzian_fit(x, y)
        self.fit_assertion(x, y, lorentzian, res)

    def test_gaussian_fit(self):

        x = [1, 11, 14, 17, 20, 28, 32]
        params = (262, 109, 18.0)
        fun = fixed_params(gaussian, *params)
        y = list(map(fun, x))
        res = gaussian_fit(x, y)
        self.fit_assertion(x, y, gaussian, res)

    def test_poly_fit(self):

        x = [5.5, 8.0, 14.0, 17.0, 35.1, 86.0]
        y = list(map(lambda x: 4 + 3 * x, x))
        res = poly_fit(x, y)
        print(res)
        self.fit_assertion(x, y, poly, res)

        x = [1.0, 2.4, 3.0, 4.1, 5.6, 6.7]
        y = list(map(lambda x: 4 - 3 * x + 2 * x**2 - x**3, x))
        res = poly_fit(x, y, n=3)
        print(res)
        self.fit_assertion(x, y, poly, res)

    def test_CSV(self):

        x = [9.4, 4.8e4, 9.3, 7.7]
        y = [44, 99, 66, 33]
        write_to_CSV(x=x, y=y)
        data = read_from_CSV('data')
        self.assertEqual(data['x'], x)
        self.assertEqual(data['y'], y)

        z = ['pico', 'de', 'gallo']
        write_to_CSV(delim=';', columnar=False, x_data=x, y_data=y, z_data=z)
        data = read_from_CSV('data', delim=';', columnar=False)
        self.assertEqual(data['x_data'], x)
        self.assertEqual(data['y_data'], y)
        self.assertEqual(data['z_data'], z)

        write_to_CSV(whats=x, a=y, seawolf=z)
        data = read_from_CSV('data')
        self.assertEqual(data['whats'], x)
        self.assertEqual(data['a'], y)
        self.assertEqual(data['seawolf'], z)