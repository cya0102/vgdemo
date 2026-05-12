"""
MoERunner_v2: Extends MoERunner with dynamic loss scheduling and multi-objective checkpointing.

New features vs MoERunner:
1. alpha_2 epoch schedule  — epoch-based diversity loss weight.
   Config field: loss.alpha_2_schedule = [[epoch_end, value], ...]
   e.g. [[4, 1.0], [10, 0.5], [30, 0.2]]  → high diversity early, tighten later.

2. aux_loss scale schedule — ramps load-balancing penalty up over epochs.
   Config field: loss.aux_loss_scale_schedule = [[epoch_end, scale], ...]
   e.g. [[4, 0.1], [10, 0.5], [30, 1.0]]  → avoid forcing uniform routing too early.

3. Multi-objective checkpoints — separately saves:
   model-best-r1.pt       (highest R@1,mIoU)
   model-best-r5.pt       (highest R@5,mIoU)
   model-best-tradeoff.pt (highest R@1 + tradeoff_r5_weight * R@5)
   Config field: tradeoff_r5_weight (default 0.35)

4. Model source — config field model.source controls which package to load from:
   "models"    -> original CPL_MoE  (default, backward compatible)
   "models_v2" -> CPL_MoE_v2        (query-conditioned shared gates)
"""

import collections
import os

import numpy as np
import torch

from models.loss import ivc_loss, cal_nll_loss, rec_loss
from runners.main_runner_moe import MoERunner, info, move_to_cuda
from utils import TimeMeter, AverageMeter


def _get_schedule_value(schedule, epoch):
    """
    Look up the value for `epoch` in a [[epoch_end, value], ...] schedule list.
    Entries must be sorted ascending by epoch_end.
    Returns the value from the first bracket whose epoch_end >= epoch,
    or the last value if epoch exceeds all brackets.
    """
    for epoch_end, value in schedule:
        if epoch <= epoch_end:
            return value
    return schedule[-1][1]


class MoERunner_v2(MoERunner):
    """
    MoERunner with epoch-based alpha_2 / aux_loss scheduling and
    three separate best-checkpoint files.
    """

    # ------------------------------------------------------------------
    # Model building — supports models_v2 via config model.source field
    # ------------------------------------------------------------------

    def _build_model(self):
        model_config = self.args["model"]
        source = model_config.get("source", "models")

        if source == "models_v2":
            import models_v2 as model_pkg
        else:
            import models as model_pkg

        self.model = getattr(model_pkg, model_config["name"])(model_config["config"])
        self.model = self.model.cuda()
        print(self.model)
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total:", total_num, "Trainable:", trainable_num)

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------

    def _get_alpha2(self, epoch: int) -> float:
        schedule = self.args["loss"].get("alpha_2_schedule", None)
        if schedule is not None:
            return _get_schedule_value(schedule, epoch)
        return self.args["loss"].get("alpha_2", 0.1)

    def _get_aux_scale(self, epoch: int) -> float:
        schedule = self.args["loss"].get("aux_loss_scale_schedule", None)
        if schedule is not None:
            return _get_schedule_value(schedule, epoch)
        return 1.0

    # ------------------------------------------------------------------
    # Training loop with multi-objective checkpointing
    # ------------------------------------------------------------------

    def train(self):
        tradeoff_r5_weight = self.args.get("tradeoff_r5_weight", 0.35)

        best_r1_results = None
        best_r5_results = None
        best_tradeoff_results = None
        best_r1_score = -1.0
        best_r5_score = -1.0
        best_tradeoff_score = -1.0

        for epoch in range(1, self.args["train"]["max_num_epochs"] + 1):
            info("Start Epoch {}".format(epoch))
            self.model_saved_path = self.args["train"]["model_saved_path"]
            os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
            save_path = os.path.join(
                self.model_saved_path, "model-{}.pt".format(epoch)
            )

            self._train_one_epoch(epoch)
            self._save_model(save_path)
            results = self.eval()

            r1 = results["R@1,mIoU"].avg
            r5 = results["R@5,mIoU"].avg
            tradeoff = r1 + tradeoff_r5_weight * r5

            if r1 > best_r1_score:
                best_r1_score = r1
                best_r1_results = results
                os.system(
                    "cp %s %s"
                    % (save_path, os.path.join(self.model_saved_path, "model-best-r1.pt"))
                )
                info("Best R@1 updated: {:.4f}".format(r1))

            if r5 > best_r5_score:
                best_r5_score = r5
                best_r5_results = results
                os.system(
                    "cp %s %s"
                    % (save_path, os.path.join(self.model_saved_path, "model-best-r5.pt"))
                )
                info("Best R@5 updated: {:.4f}".format(r5))

            if tradeoff > best_tradeoff_score:
                best_tradeoff_score = tradeoff
                best_tradeoff_results = results
                os.system(
                    "cp %s %s"
                    % (
                        save_path,
                        os.path.join(self.model_saved_path, "model-best-tradeoff.pt"),
                    )
                )
                info(
                    "Best tradeoff updated: {:.4f}  "
                    "(R@1={:.4f}, R@5={:.4f})".format(tradeoff, r1, r5)
                )

            info("=" * 60)

        info("=== Final Best Results ===")
        for label, res in [
            ("R@1-best", best_r1_results),
            ("R@5-best", best_r5_results),
            ("Tradeoff-best", best_tradeoff_results),
        ]:
            msg = "|".join(
                [" {} {:.4f} ".format(k, v.avg) for k, v in res.items()]
            )
            info("{}: |{}|".format(label, msg))

    # ------------------------------------------------------------------
    # Per-epoch training with dynamic alpha_2 and aux_loss scaling
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()

        current_alpha2 = self._get_alpha2(epoch)
        aux_scale = self._get_aux_scale(epoch)

        # Build a per-epoch loss config with the scheduled alpha_2
        loss_args = dict(self.args["loss"])
        loss_args["alpha_2"] = current_alpha2

        def print_log():
            msg = (
                "Epoch {}, Batch {}, lr = {:.5f}, "
                "alpha_2 = {:.3f}, aux_scale = {:.3f},  ".format(
                    epoch, bid, curr_lr, current_alpha2, aux_scale
                )
            )
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
                **output, num_props=self.model.num_props, **loss_args
            )
            rnk_loss, rnk_loss_dict = ivc_loss(
                **output, num_props=self.model.num_props, **loss_args
            )
            loss_dict.update(rnk_loss_dict)
            loss = loss + rnk_loss

            # aux_loss is produced by the MoE module with its built-in base weight.
            # We additionally scale it by the epoch-based aux_scale.
            aux_loss = output.get("aux_loss", None)
            if aux_loss is not None:
                scaled_aux = aux_loss * aux_scale
                loss = loss + scaled_aux
                loss_dict["aux_loss"] = aux_loss.item()
                loss_dict["aux_loss_eff"] = scaled_aux.item()

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
