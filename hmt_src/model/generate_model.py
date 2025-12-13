from copy import deepcopy

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from .language_modeling import MemoryCell, RecurrentWrapper


def generate_model(args, base_model, mem_emb_dim, logger):
    """Configure the recurrent model wrapper and return it with sequence parameters."""
    input_size = args.segment_length
    memory_size = args.mem_recall_size
    n_segments = args.bptt_depth

    if args.baseline_only:
        logger.warning(
            "training and evaluating only the backbone. remember to align the segment rightward"
        )
        memory_size = 0
        n_segments = 2

    block_size = input_size
    block_size -= 2 * memory_size
    block_size -= args.num_sensory
    history_size = (n_segments - 1) * block_size
    mask_size = block_size

    logger.info("Preparing memory cell")
    if args.rmt_only or args.baseline_only:
        cell = MemoryCell(base_model, num_mem_tokens=memory_size, num_prepend=0)

        model = RecurrentWrapper(
            cell,
            segment_size=block_size,
            max_n_segments=n_segments,
            mask_size=mask_size,
            n_cell_out=args.num_seg_save,
        )
    else:
        cell = MemoryCell(
            base_model, num_mem_tokens=memory_size, num_prepend=args.num_sensory
        )

        if args.hmt_stage_1:
            model = RecurrentWrapper(
                cell,
                segment_size=block_size,
                max_n_segments=n_segments,
                mask_size=mask_size,
                n_cell_out=args.num_seg_save,
            )
        else:
            if args.load_from_ckpt is not None and args.hmt_stage_2:
                ori_model = RecurrentWrapper(
                    cell,
                    segment_size=block_size,
                    max_n_segments=n_segments,
                    mask_size=mask_size,
                    n_cell_out=args.num_seg_save,
                )
                state_dict = get_fp32_state_dict_from_zero_checkpoint(
                    args.load_from_ckpt
                )
                ori_model.load_state_dict(state_dict)
                cell = deepcopy(ori_model.memory_cell)

            model = RecurrentWrapper(
                cell,
                mem_emb_dim=mem_emb_dim,
                hidden_dim=args.mem_hidden_dim,
                ltm_context=args.mem_queue_size,
                segment_size=block_size,
                max_n_segments=n_segments,
                mask_size=mask_size,
                n_cell_out=args.num_seg_save,
            )

    if args.load_from_ckpt is not None and not args.hmt_stage_2:
        state_dict = get_fp32_state_dict_from_zero_checkpoint(args.load_from_ckpt)
        model.load_state_dict(state_dict)

    return model, block_size, history_size
