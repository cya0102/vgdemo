"""
MoERunner_v4: runner for models_v4 / CPL_MoEv4 with Gaussian mixture proposals.

Handles:
  - model.source = "models_v4" support
  - Pull-push loss via models_v4.loss.compute_total_loss
  - aux_loss warmup schedule (entropy regularization)
  - Evaluation using center/width from mixture boundaries
"""

import collections

import numpy as np
import torch

from models_v4.loss import cal_nll_loss, compute_total_loss
from runners.main_runner_moe import MoERunner, info, move_to_cuda
from utils import TimeMeter, AverageMeter


def _get_schedule_value(schedule, epoch):
    for epoch_end, value in schedule:
        if epoch <= epoch_end:
            return value
    return schedule[-1][1]


class MoERunner_v4(MoERunner):
    def _build_model(self):
        model_config = self.args["model"]
        source = model_config.get("source", "models_v4")

        if source == "models_v4":
            import models_v4 as model_pkg
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

        loss_args = dict(self.args["loss"])
        loss_args["aux_loss_scale"] = aux_scale

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

            loss, loss_dict = compute_total_loss(
                output, num_props=self.model.num_props, loss_cfg=loss_args)

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

    def eval(self, save=None, epoch=0):
        self.model.eval()
        with torch.no_grad():
            metrics_logger = collections.defaultdict(lambda: AverageMeter())

            for bid, batch in enumerate(self.test_loader, 1):
                durations = np.asarray([i[1] for i in batch["raw"]])
                gt = np.asarray([i[2] for i in batch["raw"]])

                net_input = move_to_cuda(batch["net_input"])
                output = self.model(epoch=epoch, **net_input)
                bsz = len(durations)
                num_props = self.model.num_props
                k = min(num_props, 5)

                words_mask = output['words_mask'].unsqueeze(1) \
                    .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)
                words_id = output['words_id'].unsqueeze(1) \
                    .expand(bsz, num_props, -1).contiguous().view(bsz * num_props, -1)

                nll_loss, acc = cal_nll_loss(output['words_logit'], words_id, words_mask)
                nll_loss = nll_loss.view(bsz, num_props)
                idx = nll_loss.argsort(dim=-1)

                width = output['width'].view(bsz, num_props).gather(index=idx, dim=-1)
                center = output['center'].view(bsz, num_props).gather(index=idx, dim=-1)
                selected_props = torch.stack([
                    torch.clamp(center - width / 2, min=0),
                    torch.clamp(center + width / 2, max=1)], dim=-1)
                selected_props = selected_props.cpu().numpy()
                gt_norm = gt / durations[:, np.newaxis]

                if 'vote' in self.args and self.args['vote']:
                    if self.args['dataset']['dataset'] == 'CharadesSTA':
                        c = np.zeros((bsz, num_props))
                        for i in range(num_props):
                            iou = calculate_IoU_batch(
                                (selected_props[:, 0, 0], selected_props[:, 0, 1]),
                                (selected_props[:, i, 0], selected_props[:, i, 1]))
                            c[:, i] = iou
                    else:
                        c = np.ones((bsz, num_props))
                    votes = np.zeros((bsz, num_props))
                    for i in range(num_props):
                        for j in range(num_props):
                            iou = calculate_IoU_batch(
                                (selected_props[:, i, 0], selected_props[:, i, 1]),
                                (selected_props[:, j, 0], selected_props[:, j, 1]))
                            iou = iou * c[:, j]
                            votes[:, i] = votes[:, i] + iou
                    idx_best = np.argmax(votes, axis=1)
                    res = top_1_metric(selected_props[np.arange(bsz), idx_best], gt_norm)
                else:
                    res = top_1_metric(selected_props[:, 0], gt_norm)

                for key, v in res.items():
                    metrics_logger['R@1,' + key].update(v, bsz)
                res = top_n_metric(selected_props[:, :k].transpose(1, 0, 2), gt_norm)
                for key, v in res.items():
                    metrics_logger['R@%d,' % (k) + key].update(v, bsz)

            msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in metrics_logger.items()])
            info('|' + msg + '|')
            return metrics_logger


def calculate_IoU_batch(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def top_1_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result
