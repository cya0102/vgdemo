"""
MoERunner_v3: runner for models_v3 / CPL_MoEv3.

Extends MoERunner with:
  - model.source = "models_v3" support
  - aux_loss warmup schedule (prevents entropy regularization from dominating early)
"""

import collections

import numpy as np
import torch

from models.loss import ivc_loss, cal_nll_loss, rec_loss
from runners.main_runner_moe import MoERunner, info, move_to_cuda
from utils import TimeMeter, AverageMeter


def _get_schedule_value(schedule, epoch):
    for epoch_end, value in schedule:
        if epoch <= epoch_end:
            return value
    return schedule[-1][1]


class MoERunner_v3(MoERunner):
    def _build_model(self):
        model_config = self.args["model"]
        source = model_config.get("source", "models_v3")

        if source == "models_v3":
            import models_v3 as model_pkg
        elif source == "models_v2":
            import models_v2 as model_pkg
        else:
            import models as model_pkg

        self.model = getattr(model_pkg, model_config["name"])(model_config["config"])
        self.model = self.model.cuda()
        print(self.model)
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total:", total_num, "Trainable:", trainable_num)

    def _get_aux_scale(self, epoch):
        schedule = self.args["loss"].get("aux_loss_scale_schedule", None)
        if schedule is not None:
            return _get_schedule_value(schedule, epoch)
        return 1.0

    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()
        aux_scale = self._get_aux_scale(epoch)

        def print_log():
            msg = "Epoch {}, Batch {}, lr = {:.5f}, aux_scale = {:.3f},  ".format(
                epoch, bid, curr_lr, aux_scale)
            for k, v in loss_meter.items():
                msg += "{} = {:.4f}, ".format(k, v.avg)
                v.reset()
            msg += "{:.3f} seconds/batch".format(1.0 / time_meter.avg)
            info(msg)

        display_n_batches, bid = 50, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())

        for bid, batch in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch["net_input"])
            output = self.model(epoch=epoch, **net_input)

            loss, loss_dict = rec_loss(
                **output, num_props=self.model.num_props, **self.args['loss'])
            rnk_loss, rnk_loss_dict = ivc_loss(
                **output, num_props=self.model.num_props, **self.args['loss'])
            loss_dict.update(rnk_loss_dict)
            loss = loss + rnk_loss

            aux_loss = output.get("aux_loss", None)
            if aux_loss is not None:
                scaled_aux = aux_loss * aux_scale
                loss = loss + scaled_aux
                loss_dict["aux_loss"] = aux_loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)
            time_meter.update()
            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()
