import orbipy as op
from corrections.corrections import custom_base_correction
import numpy as np


class custom_station_keeping():
    def __init__(self,
                 model: op.models.crtbp3_model,
                 first_cor: custom_base_correction,
                 main_cor: custom_base_correction,
                 rev = np.pi / 4,
                 verbose: bool = True,
                 **kwargs):
        self.model = model
        self.first_cor = first_cor
        self.main_cor = main_cor
        self.rev = rev
        self.verbose = verbose
        self.dvout = []
        self.arr = np.array([])
        self.ev_arr = np.array([])

    def reset(self):
        self.dvout = []
        self.arr = np.array([])

    def calc_dv(self, s, cor):
        return cor.calc_zero(s) - s

    def prop(self, s0, N, ret_df=True, events=False):
        self.reset()

        t = 0
        s = s0.copy()
        if self.verbose: print('Custom station-keeping:', end=' ')
        if isinstance(self.rev, tuple):
            if events:
                det = op.event_detector(self.model, self.rev + events)
            else:
                det = op.event_detector(self.model, self.rev)
        elif events:
            det = op.event_detector(self.model, events)

        for i in range(N):
            if i == 0:
                dv = self.calc_dv(s, self.first_cor)
            else:
                try:
                    dv = self.calc_dv(s, self.main_cor)
                except Exception as e:
                    raise e
                    break
            s += dv
            self.dvout.append([t, *dv])
            if isinstance(self.rev, float):
                if events:
                    cur, ev = det.prop(s, t, t + self.rev, ret_df=False)
                else:
                    cur = self.model.prop(s, t, t + self.rev, ret_df=False)
                t += self.rev
            elif isinstance(self.rev, tuple):
                if events:
                    cur, ev = det.prop(s, t, t + np.pi, ret_df=False)
                else:
                    cur, _ = det.prop(s, t, t + np.pi, ret_df=False)
                    t = cur[-1, 0]

            s = cur[-1, 1:].copy()

            if i == 0:
                self.arr = cur[:-1].copy()
            else:
                self.arr = np.vstack((self.arr, (cur.copy() if i == N - 1 else cur[:-1].copy())))

            if events:
                if self.ev_arr.shape[0]:
                    if ev.shape[0] > 1:
                        self.ev_arr = np.vstack((self.ev_arr, ev.copy()[:-1]))
                else:
                    self.ev_arr = ev[:-1].copy()

            if self.verbose: print(i, end=' ')

        self.dvout = np.array(self.dvout)
        if ret_df:
            if events:
                return pd.DataFrame(self.arr, columns=['t', 'x', 'y', 'z', 'vx', 'vy', 'vz']), pd.DataFrame(self.ev_arr if self.ev_arr.size else None,
                                                                                                            columns=[
                                                                                                                'e',
                                                                                                                'cnt',
                                                                                                                'trm',
                                                                                                                't',
                                                                                                                'x',
                                                                                                                'y',
                                                                                                                'z',
                                                                                                                'vx',
                                                                                                                'vy',
                                                                                                                'vz'])
            else:
                return pd.DataFrame(self.arr, columns=['t', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        else:
            if events:
                return self.arr, self.ev_arr
            else:
                return self.arr


def calc_dk(df, k):
    s = 0
    N = df.shape[0]
    for i in range(N - 2 * k):
        x = np.array(
            df.loc[i, ['x', 'y', 'z', 'vx', 'vy', 'vz']] - df.loc[i + 2 * k, ['x', 'y', 'z', 'vx', 'vy', 'vz']])
        s += np.linalg.norm(x)
    return s / (N - 2 * k)