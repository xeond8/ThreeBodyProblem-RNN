import time

import orbipy as op
import numpy as np
from scipy.optimize import bisect
import torch
import pandas as pd
import matplotlib.pyplot as plt

from ml.lstm import LSTMModel, FC_Class, LSTMClassifier, NewOldLSTMClassifier, GRUClassifier, CNNLSTMClassifier, \
    OldLSTMClassifier


class custom_base_correction():
    def __init__(self,
                 sign_evaluator,
                 direction: op.directions.base_direction = op.y_direction(),
                 param_init: float = 0,
                 param_delta: float = np.pi / 9,
                 model: op.models.crtbp3_model = op.crtbp3_model(),
                 param_n_points: int = 20,
                 tol: float = 1e-10,
                 max_iter_bisect: int = 30,
                 max_iter: int = 5,
                 maxt: float = 6 * np.pi,
                 Nt: int = 60,
                 T: float = 2 * np.pi,
                 skipt: float = 0,
                 normalise=False):
        self.sign_evaluator = sign_evaluator
        self.direction = direction
        self.param_init = param_init
        self.param_delta = param_delta
        self.model = model
        self.param_n_points = param_n_points
        self.tol = tol
        self.max_iter = max_iter
        self.max_iter_bisect = max_iter_bisect
        self.maxt = maxt
        self.Nt = Nt
        self.T = T
        self.skipt = skipt
        self.vec_direction = None
        self.normalise = normalise

    def get_vec(self, param, s):
        temp = s.copy()
        if self.vec_direction is not None:
            temp[3:] += param * self.vec_direction
        else:
            temp[3:] += param * self.direction(0, s)
        return temp

    def get_range(self, n):
        return self.param_init + np.linspace(-self.param_delta, self.param_delta, self.param_n_points * (n + 1))

    def get_range_last(self, n):
        arrl = self.param_init + np.linspace(-3 * self.param_delta, -self.param_delta, n * self.param_n_points)
        arrr = self.param_init + np.linspace(self.param_delta, 3 * self.param_delta, n * self.param_n_points)
        return np.concatenate([arrl, arrr])

    def get_sign(self, param, s):
        if type(self.sign_evaluator) == op.event_detector:
            detector = self.sign_evaluator
            vec = self.get_vec(param, s)
            if vec is None:
                return 0
            df, ev = detector.prop(vec, 0, self.maxt, ret_df=False)
            if ev.shape[0] < 1:
                return 0
            else:
                return -1 if ev[-1, 0] < 1 else 1

        if isinstance(self.sign_evaluator, FC_Class):
            fc_model = self.sign_evaluator
            fc_model.eval()
            vec = self.get_vec(param, s)
            if vec is None:
                return 0

            mini_X = np.zeros((self.Nt, 6))
            mini_X[0] = vec

            if self.skipt:
                df_prop = self.model.prop(vec, 0, self.skipt)
                vec = np.array(df_prop.iloc[-1, 1:])

            for k in range(1, self.Nt):
                df_prop = self.model.prop(vec, self.T / self.Nt * (k - 1), self.T / self.Nt * k)
                vec = np.array(df_prop.iloc[-1, 1:])
                mini_X[k] = vec

            mini_X = torch.tensor(mini_X.reshape(1, self.Nt * 6), dtype=torch.float32)
            if self.normalise:
                mini_X[:, ::6] -= self.model.L1
            res = fc_model(mini_X)
            res = res.squeeze(-1)
            return 1 if res > 1 / 2 else -1

        elif isinstance(self.sign_evaluator, OldLSTMClassifier) or isinstance(self.sign_evaluator, NewOldLSTMClassifier) or self.sign_evaluator.pooling_type == "None":

            lstm_model = self.sign_evaluator
            lstm_model.eval()
            vec = self.get_vec(param, s)
            if vec is None:
                return 0
            mini_X = np.zeros((self.Nt, 6))
            mini_X[0] = vec

            if self.skipt:
                df_prop = self.model.prop(vec, 0, self.skipt)
                vec = np.array(df_prop.iloc[-1, 1:])

            for k in range(1, self.Nt):
                df_prop = self.model.prop(vec, self.T / self.Nt * (k - 1), self.T / self.Nt * k)
                vec = np.array(df_prop.iloc[-1, 1:])
                mini_X[k] = vec

            mini_X = torch.tensor(mini_X.reshape(1, self.Nt, 6), dtype=torch.float32)
            if self.normalise:
                mini_X[:, :, 0] -= self.model.L1
            res = lstm_model(mini_X)

            if isinstance(self.sign_evaluator, OldLSTMClassifier):
                return 1 if res.squeeze(-1)[:, -1].item() > 1 / 2 else -1
            else:
                return 1 if res.squeeze(-1)[:, -1].item() > 0 else -1
                return 1 if res[0].item() > 1 / 2 else -1


        elif isinstance(self.sign_evaluator, GRUClassifier) or isinstance(self.sign_evaluator, CNNLSTMClassifier) or isinstance(self.sign_evaluator, LSTMClassifier):
            gru_model = self.sign_evaluator
            gru_model.eval()
            vec = self.get_vec(param, s)
            if vec is None:
                return 0
            mini_X = np.zeros((self.Nt, 6))
            mini_X[0] = vec

            if self.skipt:
                df_prop = self.model.prop(vec, 0, self.skipt)
                vec = np.array(df_prop.iloc[-1, 1:])

            for k in range(1, self.Nt):
                df_prop = self.model.prop(vec, self.T / self.Nt * (k - 1), self.T / self.Nt * k)
                vec = np.array(df_prop.iloc[-1, 1:])
                mini_X[k] = vec
                #if vec[0] < 0.97:
                    #break

            mini_X = torch.tensor(mini_X.reshape(1, self.Nt, 6), dtype=torch.float32)
            lengths = torch.tensor([k], dtype=torch.long)
            if self.normalise:
                mini_X[:, :, 0] -= self.model.L1
            #res = gru_model(mini_X, lengths)
            res = gru_model(mini_X)


            return 1 if res.item() > 0 else -1

    def calc_zero(self, s):
        self.vec_direction = self.direction(0, s)
        for n in range(self.max_iter):
            if n < self.max_iter - 1:
                param_range = self.get_range(n)
            else:
                param_range = self.get_range_last(n)
            sign_arr = np.array([self.get_sign(param, s) for param in param_range])
            gaps = (sign_arr[:-1] * sign_arr[1:]) < 0
            n_gaps = sum(gaps)
            if n_gaps >= 1:
                gaps_ind = np.where(gaps)[0]
                i = gaps_ind[np.argmin(np.abs(gaps_ind - len(gaps) // 2))]
                # i = gaps_ind[0]
                l, r = param_range[i], param_range[i + 1]
                ans = bisect(self.get_sign, l, r, args=(s,), maxiter=self.max_iter_bisect, xtol=self.tol)
                res = self.get_vec(ans, s)
                self.vec_direction = None
                return res
            '''elif n_gaps > 1:
                raise ValueError('Multiple zeros')'''
        raise ValueError('No zeros')


class zero_velocity_correction(custom_base_correction):
    def __init__(self,
                 r: float,
                 angle_initial: float = np.pi / 2,
                 angle_delta: float = np.pi / 9,
                 angle_n_points: int = 6,
                 *args, **kwargs
                 ):
        super().__init__(
            param_init=angle_initial,
            param_delta=angle_delta,
            param_n_points=angle_n_points,
            *args, **kwargs
        )
        self.r = r
        self.angle_initial = angle_initial
        self.angle_delta = angle_delta
        self.angle_n_points = angle_n_points

    def get_vec(self, param, s):
        temp = s.copy()
        temp[[0, 2]] += self.r * np.array([np.cos(param), np.sin(param)])
        return temp

    def get_zvc(self, n_points=100, s=None):
        if s is None:
            s = np.array([self.model.L1, 0., 0., 0., 0., 0.])
        df = pd.DataFrame(0., index=np.arange(n_points), columns=['t', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        df.iloc[0, 1:] = s

        for i in range(1, n_points):
            try:
                s = self.calc_zero(s)
                df.iloc[i, 1:] = s.copy()
                self.param_init = np.atan((df.iloc[i, 3] - df.iloc[i - 1, 3]) / (df.iloc[i, 1] - df.iloc[i - 1, 1]))
            except Exception as e:
                print(e)
                print(s)
                return df.iloc[:i]

        return df


class jacobi_isoline_correction(custom_base_correction):
    def __init__(self,
                 r: float,
                 v_sign: float = 1.,
                 angle_initial: float = np.pi,
                 angle_delta: float = np.pi / 6,
                 angle_n_points: int = 6,
                 *args, **kwargs):
        super().__init__(
            param_init=angle_initial,
            param_delta=angle_delta,
            param_n_points=angle_n_points,
            *args, **kwargs
        )
        self.r = r
        self.v_sign = v_sign
        self.angle_initial = angle_initial
        self.angle_delta = angle_delta
        self.angle_n_points = angle_n_points

    def get_vec(self, param, s):
        temp = s.copy()
        temp[[0, 2]] += self.r * np.array([np.cos(param), np.sin(param)])
        v2 = 2 * self.model.omega(temp) - self.model.jacobi(s)
        if v2 < 0:
            return None
        if self.vec_direction is not None:
            temp[3:] = self.v_sign * np.sqrt(v2) * self.vec_direction
        else:
            temp[3:] = self.v_sign * np.sqrt(v2) * self.direction(0, s)
        return temp

    def get_isoline(self, s0, alphas=False, oneside=False, early_stop=1000):
        self.param_init = self.angle_initial
        x_arr = [s0[0]]
        z_arr = [s0[2]]
        vy_arr = [s0[4]]
        s = s0.copy()
        k = 0
        while s[2] >= 0 and k < early_stop:
            try:
                if k <= 1:
                    self.param_delta *= 5
                    s = self.calc_zero(s)
                    self.param_delta /= 5
                else:
                    s = self.calc_zero(s)
                x_arr.append(s[0])
                z_arr.append(s[2])
                vy_arr.append(s[4])
                if alphas:
                    if len(x_arr) >= 3:
                        alpha2 = np.atan2((z_arr[-1] - z_arr[-2]), (x_arr[-1] - x_arr[-2]))
                        alpha1 = np.atan2((z_arr[-2] - z_arr[-3]), (x_arr[-2] - x_arr[-3]))
                        self.param_init = 2 * alpha2 - alpha1
                    else:
                        self.param_init = np.atan2((z_arr[-1] - z_arr[-2]), (x_arr[-1] - x_arr[-2]))
                else:
                    self.param_init = np.atan2((z_arr[-1] - z_arr[-2]), (x_arr[-1] - x_arr[-2]))

                k += 1
            except Exception as e:
                print(e)
                break

        if oneside:
            df = pd.DataFrame(0., index=np.arange(len(x_arr), dtype=int),
                              columns=['t', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
            df['x'] = x_arr
            df['z'] = z_arr
            df['vy'] = vy_arr
            return df

        x_arr = x_arr[::-1]
        z_arr = z_arr[::-1]
        vy_arr = vy_arr[::-1]
        s = s0.copy()
        self.param_init = self.angle_initial - np.pi
        self.v_sign *= -1.

        flag = False
        k = 0
        while s[2] >= 0 and k < 300:
            try:
                if k <= 1:
                    self.param_delta *= 5
                    s = self.calc_zero(s)
                    self.param_delta /= 5
                else:
                    s = self.calc_zero(s)
                x_arr.append(s[0])
                z_arr.append(s[2])
                vy_arr.append(s[4])
                if alphas:
                    if flag:
                        alpha2 = np.atan2((z_arr[-1] - z_arr[-2]), (x_arr[-1] - x_arr[-2]))
                        alpha1 = np.atan2((z_arr[-2] - z_arr[-3]), (x_arr[-2] - x_arr[-3]))
                        self.param_init = 2 * alpha2 - alpha1
                    else:
                        self.param_init = np.atan2((z_arr[-1] - z_arr[-2]), (x_arr[-1] - x_arr[-2]))
                        flag = True
                else:
                    self.param_init = np.atan2((z_arr[-1] - z_arr[-2]), (x_arr[-1] - x_arr[-2]))

                k += 1
            except Exception as e:
                break

        df = pd.DataFrame(0., index=np.arange(len(x_arr), dtype=int), columns=['t', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        df['x'] = x_arr
        df['z'] = z_arr
        df['vy'] = vy_arr

        self.v_sign *= -1
        return df



class lyapunov_correction(custom_base_correction):
    def __init__(self,
                 r: float,
                 v_init: float = 0,
                 v_delta: float = 0.00001,
                 v_n_points: int = 6,
                 *args, **kwargs):
        super().__init__(
            param_init=v_init,
            param_delta=v_delta,
            param_n_points=v_n_points,
            *args, **kwargs
        )
        self.r = r
        self.v_n_points = v_n_points
        self.v_init = v_init
        self.v_delta = v_delta
        self.dx_sign = 1.

    def get_vec(self, param, s):
        temp = s.copy()
        temp[0] += self.dx_sign * self.r
        temp[4] += param
        return temp

    def get_sign(self, param, s):
        if self.sign_evaluator:
            lstm_model = self.sign_evaluator
            lstm_model.eval()
            vec = self.get_vec(param, s)
            if vec is None:
                return 0
            mini_X = np.zeros((self.Nt, 6))
            mini_X[0] = vec

            if self.skipt:
                df_prop = self.model.prop(vec, 0, self.skipt)
                vec = np.array(df_prop.iloc[-1, 1:])

            for k in range(1, self.Nt):
                df_prop = self.model.prop(vec, self.T / self.Nt * (k - 1), self.T / self.Nt * k)
                vec = np.array(df_prop.iloc[-1, 1:])
                mini_X[k] = vec
            mini_X = torch.tensor(mini_X.reshape(1, self.Nt, 6), dtype=torch.float32)
            if self.normalise:
                mini_X[:, :, 0] -= self.model.L1
            res = lstm_model(mini_X)
            if isinstance(self.sign_evaluator, LSTMModel):
                return 1 if res[0].squeeze(-1)[:, -1].item() > 0 else -1
            else:
                return 1 if res[0].squeeze(-1)[:, -1].item() > 1 / 2 else -1
        else:
            vec = self.get_vec(param, s)
            if vec is None:
                return 0
            return self.model.prop(vec, 0, np.pi).iloc[-1, 1] - vec[0]

    def get_loc(self, n_points=[50, 50], s=None):
        if s is None:
            s = np.array([self.model.L1, 0., 0., 0., 0., 0.])

        x_arr = [s[0]]
        vy_arr = [s[4]]
        for i in range(n_points[0]):
            try:
                s = self.calc_zero(s)
                x_arr.append(s[0])
                vy_arr.append(s[4])
                self.param_init = vy_arr[-1] - vy_arr[-2]
            except Exception as e:
                print(e)
                print(s)
                break

        x_arr = x_arr[::-1]
        vy_arr = vy_arr[::-1]
        self.dx_sign = -1.
        s = np.array([x_arr[-1], 0., 0., 0., vy_arr[-1], 0])
        self.param_init = vy_arr[-1] - vy_arr[-2]
        for i in range(n_points[1]):
            try:
                s = self.calc_zero(s)
                x_arr.append(s[0])
                vy_arr.append(s[4])
                self.param_init = vy_arr[-1] - vy_arr[-2]
            except Exception as e:
                print(e)
                print(s)
                break

        df = pd.DataFrame(0., index=np.arange(len(x_arr)), columns=['t', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        df['x'] = x_arr
        df['vy'] = vy_arr
        return df


class velocity_correction(custom_base_correction):
    def __init__(self,
                 velocity_initial: float = 0.,
                 velocity_delta: float = 0.01,
                 velocity_n_points: int = 10,
                 *args, **kwargs):
        super().__init__(
            param_init=velocity_initial,
            param_delta=velocity_delta,
            param_n_points=velocity_n_points,
            *args, **kwargs
        )
        self.velocity_initial = velocity_initial
        self.velocity_delta = velocity_delta
        self.velocity_n_points = velocity_n_points

    def get_vec(self, param, s):
        temp = s.copy()
        if self.vec_direction is not None:
            temp[3:] = param * self.vec_direction
        else:
            temp[3:] = param * self.direction(0, s)
        return temp


class zero_correction(custom_base_correction):
    def __init__(self):
        pass

    def calc_zero(self, s: np.ndarray):
        return s

