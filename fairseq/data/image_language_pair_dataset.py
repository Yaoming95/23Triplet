# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
import random
import copy

logger = logging.getLogger(__name__)

PAD_LEN = 512
SEP_TOKEN = "[U_0]"
EOS_TOKEN = "</s>"
PLHD_TOKEN = "[U_1]"
MASK_TOKEN = "[U_2]"

colors = ['orange', 'green', 'red', 'white', 'black', 'pink', 'blue', 'purple', 'tan', 'grey', 'gray', 'yellow', 'gold',
          'golden', 'dark', 'brown', 'silver']


def get_color_set(fairseq_dict):
    color_id_list = []
    mask_c_id = None
    for id in range(len(fairseq_dict)):
        if fairseq_dict[id] in colors:
            color_id_list.append(id)
        if fairseq_dict[id] == "[MASK_C]":
            mask_c_id = id

    # assert len(colors)==len(color_id_list)
    assert mask_c_id is not None
    return color_id_list, mask_c_id


def get_sep_id(fairseq_dict):
    for id in range(len(fairseq_dict) - 1, 0 - 1, -1):
        if fairseq_dict[id] == SEP_TOKEN:
            return id
    return len(fairseq_dict) - 1


def get_plhd_id(fairseq_dict):
    for id in range(len(fairseq_dict) - 1, 0 - 1, -1):
        if fairseq_dict[id] == PLHD_TOKEN:
            return id
    return len(fairseq_dict) - 2


def get_mask_id(fairseq_dict):
    for id in range(len(fairseq_dict) - 1, 0 - 1, -1):
        if fairseq_dict[id] == MASK_TOKEN:
            return id
    return len(fairseq_dict) - 3


def get_eos_id(fairseq_dict):
    for id in range(len(fairseq_dict) - 1, 0 - 1, -1):
        if fairseq_dict[id] == EOS_TOKEN:
            return id
    return len(fairseq_dict) - 3


def collate(
        samples,
        pad_idx,
        eos_idx,
        left_pad_source=True,
        left_pad_target=False,
        input_feeding=True,
        pad_to_length=None,
        pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
                alignment[:, 0].max().item() >= src_len - 1
                or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    # extra code
    imgs_list = [[] for i in range(len(samples[0]["img_list"]))]
    for s in samples:
        for idx, img in enumerate(s["img_list"]):
            imgs_list[idx].append(img)
    for idx, i in enumerate(imgs_list):
        img = torch.stack(i, dim=0)
        img = img.index_select(0, sort_order)
        imgs_list[idx] = img

    img_masks_list = []
    img_masks_pos = []
    for idx, i in enumerate(samples[0]["img_mask_list"]):
        if i is not None:
            img_masks_list.append([])
            img_masks_pos.append(idx)
        else:
            img_masks_list.append(None)
    for s in samples:
        for idx, img_mask in enumerate(s["img_mask_list"]):
            if idx in img_masks_pos:
                img_masks_list[idx].append(img_mask)
    for idx, i in enumerate(img_masks_list):
        if idx in img_masks_pos:
            img_mask = torch.stack(i, dim=0)
            img_mask = img_mask.index_select(0, sort_order)
            img_masks_list[idx] = img_mask

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "imgs_list": imgs_list,  #
            "img_masks_list": img_masks_list,  #
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    if samples[0].get("alignment", None) is not None:
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0: lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints

    return batch


class ImageLanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        imgs (ImageDataset): list for image dataset
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
            self,
            src,
            src_sizes,
            src_dict,
            imgs,
            tgt=None,
            tgt_sizes=None,
            tgt_dict=None,
            left_pad_source=True,
            left_pad_target=False,
            shuffle=True,
            input_feeding=True,
            remove_eos_from_source=False,
            append_eos_to_target=False,
            align_dataset=None,
            constraints=None,
            append_bos=False,
            eos=None,
            num_buckets=0,
            src_lang_id=None,
            tgt_lang_id=None,
            pad_to_multiple=1,
            args=None,
            split="train",
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                    self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        # self.args = args
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple
        self.imgs = imgs  # extra code

        self.target_prompt = 0.0
        get_color_set(self.tgt_dict)
        self.image_mask = 0.0
        self.split = split

        if args is not None:
            self.target_prompt = args.get("target_prompt", 0.0)
            if self.target_prompt > 0.0:
                self.sep_id = get_sep_id(self.src_dict)
            self.dynamic_mask_color = args.get("dynamic_mask_color", 0.0)
            if self.dynamic_mask_color > 0.0:
                self.color_id_list, self.mask_c_id = get_color_set(self.src_dict)
            self.image_mask = args.get("image_mask", 0.0)
            self.prompt_mask = args.get("prompt_mask", 0.0)
            if self.image_mask or self.prompt_mask:
                self.sep_id = get_sep_id(self.src_dict)
            if split != "train":
                self.dynamic_mask_color = 0.0
                self.target_prompt = 0.0
                self.image_mask = 0.0
                self.prompt_mask = 0.0

        self.plhd_id = get_plhd_id(self.src_dict)
        self.mask_id = get_mask_id(self.src_dict)
        self.eos_id = get_eos_id(self.src_dict)

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]

        img_item = [i[index][0] for i in self.imgs]  # list for image data
        img_mask_item = [i[index][1] for i in self.imgs]  # list for image mask data

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        if self.dynamic_mask_color:
            src_item = copy.deepcopy(src_item)
            for id in range(len(src_item)):
                if src_item[id] in self.color_id_list:
                    if random.random() < self.dynamic_mask_color:
                        src_item[id] = self.mask_c_id

        drop_num = random.random()
        if self.prompt_mask:
            if self.sep_id in src_item and bool(img_item[0].sum()):
                if self.prompt_mask >= 1 or drop_num < self.prompt_mask:
                    sep_loc = (src_item == self.sep_id).nonzero().item()
                    src_item = torch.cat([src_item[:sep_loc], src_item[-1:]])

        if self.image_mask:
            if src_item[0] != self.plhd_id:
                if self.image_mask >= 1 or drop_num < self.image_mask:
                    img_item = [torch.tensor([0.0])]

        if self.target_prompt:
            if self.sep_id in src_item:
                src_item = copy.deepcopy(src_item)
                prompt_start_idx = (src_item == self.sep_id).nonzero().item()
                prompt_end_idx = len(src_item) - 1

                prompts = src_item[prompt_start_idx + 1:prompt_end_idx]
                prompts = copy.deepcopy(prompts)
                # replace
                for idx in range(len(prompts)):
                    if random.random() < 1 / 5:
                        prompts[idx] = random.randint(4, len(self.src_dict) - 20)
                rand_num = random.random()
                # random add
                if rand_num < 1 / 3:
                    change_num = torch.distributions.binomial.Binomial(len(prompts), 1 / 5).sample().int()
                    new_id = [random.randint(4, len(self.src_dict) - 20) for x in range(change_num)]
                    prompts = torch.cat([prompts, torch.Tensor(new_id).int()], dim=0)
                    ids = torch.randperm(len(prompts))
                    prompts = prompts[ids]
                # random delete
                elif 1 / 3 < rand_num < 2 / 3:
                    change_num = torch.distributions.binomial.Binomial(len(prompts), 1 / 5).sample().int()
                    if change_num and change_num < len(prompts):
                        prompts = prompts[:-change_num]

                # shuffle prompts
                rand_num = random.random()
                if rand_num < 1 / 3:
                    ids = torch.randperm(len(prompts))
                    prompts = prompts[ids]
                src_item = torch.cat([src_item[:prompt_start_idx + 1], prompts, src_item[-1:]])




            else:
                src_item = copy.deepcopy(src_item)
                if random.random() < self.target_prompt and self.sep_id not in src_item:
                    ids = torch.randperm(len(tgt_item[:-1]))
                    prompts = tgt_item[:-1][ids]
                    for idx, token in enumerate(prompts):
                        rand_num = random.random()
                        if rand_num < 1 / 5:
                            prompts[idx] = random.randint(4, len(self.src_dict) - 20)
                    valid_id_num = torch.distributions.binomial.Binomial(len(prompts), 1 / 6).sample().int()

                    valid_id_num = max(valid_id_num, 1)
                    prompts = prompts[:valid_id_num]

                    src_item = torch.cat([src_item[:-1], torch.tensor([self.sep_id]), prompts, src_item[-1:]])

        def random_mask(item, ):
            for id in range(len(item) - 1):
                if random.random() < 0.35:
                    item[id] = self.mask_id
            return item

        if src_item[0] == self.plhd_id:
            src_item = torch.cat([src_item[0].unsqueeze(0), random_mask(copy.deepcopy(tgt_item))])
        # [self.src_dict[x] for x in src_item]
        # [self.src_dict[x] for x in tgt_item]

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            "img_list": img_item,
            "img_mask_list": img_mask_item,
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `imgs_list` (List): a list for LongTensor, a 3D Tensor of image
                  - `img_masks_list` (List): a list for LongTensor, a 2D Tensor of image_mask or None
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        for item in samples:
            # pad if is None
            if torch.numel(item["img_list"][0]) <= 1:
                item["img_list"][0] = torch.zeros([1, PAD_LEN])
            if len(item["img_list"][0].shape) == 1:
                item["img_list"] = [torch.unsqueeze(x, 0) for x in item["img_list"]]

        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )

        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        # if type(res["net_input"]["imgs_list"]) is list:
        #     res["net_input"]["imgs_list"] = torch.stack(res["net_input"]["imgs_list"])
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
                getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
