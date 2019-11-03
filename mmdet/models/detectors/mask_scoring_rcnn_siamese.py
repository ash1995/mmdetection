import torch

import mmcv
import numpy as np
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, tensor2imgs
from .. import builder
from ..registry import DETECTORS
from .two_stage import TwoStageDetector

@DETECTORS.register_module
class SiameseMaskScoringRCNN(TwoStageDetector):
    """Mask Scoring RCNN.

    https://arxiv.org/abs/1903.00241
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 mask_iou_head=None,
                 pretrained=None):
        super(SiameseMaskScoringRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.mask_iou_head = builder.build_head(mask_iou_head)
        self.mask_iou_head.init_weights()

    def forward_dummy(self, img):
        raise NotImplementedError

    # TODO: refactor forward_train in two stage to reduce code redundancy
    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        img1 = img[0]
        img2 = img[1]

        x1 = self.extract_feat(img1)
        x2 = self.extract_feat(img2)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x1)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img1.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x1])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x2[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x2[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks,
                                                     self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

            # mask iou head forward and loss
            pos_mask_pred = mask_pred[range(mask_pred.size(0)), pos_labels]
            mask_iou_pred = self.mask_iou_head(mask_feats, pos_mask_pred)
            pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)
                                                    ), pos_labels]
            mask_iou_targets = self.mask_iou_head.get_target(
                sampling_results, gt_masks, pos_mask_pred, mask_targets,
                self.train_cfg.rcnn)
            loss_mask_iou = self.mask_iou_head.loss(pos_mask_iou_pred,
                                                    mask_iou_targets)
            losses.update(loss_mask_iou)
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):

        num_augs = len(imgs) / 2
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0][0].size(0)
        assert imgs_per_gpu == 1


        if num_augs == 1:
            return self.simple_test(imgs, img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x1 = self.extract_feat(img[0][0])
        x2 = self.extract_feat(img[1][0])

        proposal_list = self.simple_test_rpn(
            x1, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x2, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x2, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
            mask_scores = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes,
                                                       det_labels,
                                                       self.test_cfg.rcnn,
                                                       ori_shape, scale_factor,
                                                       rescale)
            # get mask scores with mask iou head
            mask_iou_pred = self.mask_iou_head(
                mask_feats,
                mask_pred[range(det_labels.size(0)), det_labels + 1])
            mask_scores = self.mask_iou_head.get_mask_scores(
                mask_iou_pred, det_bboxes, det_labels)
        return segm_result, mask_scores


    def show_result(self, data, result, dataset=None, score_thr=0.3):

      if isinstance(result, tuple):
          bbox_result, segm_result = result
      else:
          bbox_result, segm_result = result, None

      img_tensor = data['img'][1][0]
      img_metas = data['img_meta'][0].data[0]
      #import pdb
      #pdb.set_trace()
      imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
      assert len(imgs) == len(img_metas)

      if dataset is None:
          class_names = self.CLASSES
      elif isinstance(dataset, str):
          class_names = get_classes(dataset)
      elif isinstance(dataset, (list, tuple)):
          class_names = dataset
      else:
          raise TypeError(
              'dataset must be a valid dataset name or a sequence'
              ' of class names, not {}'.format(type(dataset)))

      for img, img_meta in zip(imgs, img_metas):
          h, w, _ = img_meta['img_shape']
          img_show = img[:h, :w, :]

          bboxes = np.vstack(bbox_result)
          # draw segmentation masks
          if segm_result is not None:
              segms = mmcv.concat_list(segm_result)
              inds = np.where(bboxes[:, -1] > score_thr)[0]
              for i in inds:
                  color_mask = np.random.randint(
                      0, 256, (1, 3), dtype=np.uint8)
                  mask = maskUtils.decode(segms[i]).astype(np.bool)
                  img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
          # draw bounding boxes
          labels = [
              np.full(bbox.shape[0], i, dtype=np.int32)
              for i, bbox in enumerate(bbox_result)
          ]
          labels = np.concatenate(labels)
          mmcv.imshow_det_bboxes(
              img_show,
              bboxes,
              labels,
              class_names=class_names,
              score_thr=score_thr)
