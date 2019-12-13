import os
from timeit import default_timer as timer

import numpy as np

from deepartransit.utils.transit import LinearTransit
from .deeparsys import DeepARSysModel, DeepARSysTrainer

"""
Variant from deepAR original network, adapted to transit light curve structure
"""


class DeepARTransModel(DeepARSysModel):
    def __init__(self, config):
        super().__init__(config)


class DeepARTransTrainer(DeepARSysTrainer):
    def __init__(self, sess, model, data, config, logger=None, transit_model=LinearTransit):
        super().__init__(sess, model, data, config, logger=logger)
        self.transit = [[]] * self.data.Z.shape[0]
        for i in range(self.data.Z.shape[0]):
            self.transit[i] = transit_model(data.time_array[:self.model.T])

    def eval_step(self, verbose=True):
        cur_it = self.model.global_step_tensor.eval()
        if self.config.num_features > 10:
            weights = self.data.scaler_Z.centers.squeeze(0) / self.data.scaler_Z.centers.mean(axis=-1).squeeze(0)
        else:
            weights = np.ones(self.data.Z[:, 0, :].shape)

        feed_dict = {self.model.Z: self.data.Z, self.model.X: self.data.X, self.model.weights: weights}

        t1 = timer()
        locs, scales = self.sess.run([self.model.loc_at_time, self.model.scale_at_time], feed_dict=feed_dict)
        np.save(os.path.join(self.config.output_dir, 'loc_array_{}.npy'.format(cur_it)), np.array(locs))
        np.save(os.path.join(self.config.output_dir, 'scales_array_{}.npy'.format(cur_it)), np.array(scales))
        t2 = timer()

        # Fit transit
        #################
        transit_component = (self.data.scaler_Z.inverse_transform(self.data.Z[:, :self.model.T]).sum(-1) /
                             self.data.scaler_Z.inverse_transform(np.swapaxes(locs, 0, 1)).sum(-1))
        for i in range(self.data.Z.shape[0]):
            try:
                B = min(self.model.T // 20, 5)  # ensure we have at least 90% of the points for the fit
                p0 = None if (self.transit[i].transit_pars is None or
                              np.isinf(np.array(self.transit[i].transit_pars)).any()
                              or np.isnan(self.transit[i].transit_pars).any()) else self.transit[i].transit_pars
                self.transit[i].fit(transit_component[i], range_fit=range(0 + B, self.model.T - B),
                                    p0=p0, time_axis=0)
            except NotImplementedError:
                print('problem when fitting (ValueError)')
                continue
            print('delta:', self.transit[i].delta)

        t4 = timer()

        # compute metrics
        ###############
        pred_range = range(self.config.pretrans_length, self.config.pretrans_length + self.config.trans_length)
        transit_fit = np.array([self.transit[i].flux for i in range(self.data.Z.shape[0])])
        mse_transit = ((transit_component - transit_fit)[:, pred_range] ** 2).mean()
        std_depths = np.std([self.transit[i].delta for i in range(len(self.transit))])
        # TENSORBOARD eval summary
        lr = self.model.learning_rate_tensor.eval()
        summaries_dict = {
            'mse_transit': np.array(mse_transit),
            'std_transit': np.array(std_depths),
            'learning_rate': lr
        }
        self.logger.summarize(cur_it, summarizer='test', summaries_dict=summaries_dict)

        if self.config.early_stop:
            if summaries_dict[self.config.early_stop_metric] < self.best_score:
                self.model.save(self.sess)
                self.best_score = summaries_dict[self.config.early_stop_metric]
                self.accum_early = 0
            else:
                self.accum_early += 1

        return timer() - t1, summaries_dict

    def update_ranges(self, margin=1.05, verbose=True):
        for obs in range(self.data.Z.shape[0]):
            if np.isnan(self.transit[obs].duration) or np.isinf(self.transit[obs].duration):
                duration = self.model.config['trans_length']
            else:
                duration = self.transit[obs].duration
            self.model.trans_length[obs] = int(np.ceil(duration * margin))
            self.model.pretrans_length[obs] = int(np.floor(self.transit[obs].t_c - self.model.trans_length[obs] // 2))
            self.model.postrans_length[obs] = (
                        self.model.T - (self.model.trans_length[obs] + self.model.pretrans_length[obs]))
            self.model.margin_length[obs] = (self.model.trans_length[obs] - self.transit[obs].duration) // 2
            if verbose:
                print('Transit length recomputed with margin {}: {}'.format(margin, self.model.trans_length))
