import numpy as np
import pandas as pd
from corrections.corrections import custom_base_correction

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

    def get_isoline(self, s0, alphas=False, oneside=False, early_stop=1000, change_sign=True):
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
        if change_sign:
            self.v_sign *= -1.

        flag = False
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
