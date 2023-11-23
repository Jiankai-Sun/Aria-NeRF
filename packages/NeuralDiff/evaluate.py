import argparse
import os

import numpy as np

import evaluation
import utils
from dataset import SAMPLE_IDS, VIDEO_IDS, EPICDiff, MaskLoader


def parse_args(path=None, vid=None, exp=None):

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default=path, help="Path to model.")

    parser.add_argument("--vid", type=str, default=vid, help="Video ID of dataset.")

    parser.add_argument("--exp", type=str, default=exp, help="Experiment name.")

    parser.add_argument(
        "--outputs",
        default=["masks"],
        type=str,
        nargs="+",
        help="Evaluation output. Select `masks` or `summary` or both.",
    )

    parser.add_argument(
        "--masks_n_samples",
        type=int,
        default=0,
        help="Select number of samples for evaluation. If kept at 0, then all test samples are evaluated.",
    )

    parser.add_argument(
        "--summary_n_samples",
        type=int,
        default=0,
        help="Number of samples to evaluate for summary video. If 0 is selected, then the video is rendered with all frames from the dataset.",
    )

    parser.add_argument(
        "--root_data", type=str, default="data/EPIC-Diff", help="Root of the dataset."
    )

    parser.add_argument(
        "--suppress_person",
        default=False,
        action="store_true",
        help="Disables person, e.g. for visualising complete foreground without parts missing where person occludes the foreground.",
    )

    # metrics
    parser.add_argument(
        "--use_lpips",
        type=int,
        default=0,
        help="use lpips metric",
    )
    parser.add_argument(
        "--use_ssim",
        type=int,
        default=0,
        help="use ssim metric",
    )

    # for opt.py
    parser.add_argument("--is_eval_script", default=True, action="store_true")

    args = parser.parse_args()

    return args


def init(args):

    dataset = EPICDiff(args.vid, root=args.root_data)

    model = utils.init_model(args.path, dataset)

    # update parameters of loaded models
    model.hparams["suppress_person"] = args.suppress_person
    model.hparams["inference"] = True

    return model, dataset


def eval_masks(args, model, dataset, root):
    """Evaluate masks to produce mAP (and PSNR) scores."""
    root = os.path.join(root, "masks")
    os.makedirs(root, exist_ok=True)
    try:
        maskloader = MaskLoader(dataset=dataset)
    except:
        print('Mask does not exist. Skip.')
        maskloader = None

    image_ids = evaluation.utils.sample_linear(
        dataset.img_ids_test, args.masks_n_samples
    )[0]

    results = evaluation.evaluate(
        dataset,
        model,
        maskloader,
        vis_i=1,
        save_dir=root,
        save=True,
        vid=args.vid,
        image_ids=image_ids,
        use_lpips=args.use_lpips,
        use_ssim=args.use_ssim,
    )
    # print('results: ', results)
    # avgpre, masks, mask_tran, im_stat, im_targ, psnr, mask_pred


def eval_masks_average(args):
    """Calculate average of `eval_masks` results for all 10 scenes."""
    scores = []
    for vid in VIDEO_IDS:
        path_metrics = os.path.join("results", args.exp, vid, 'masks', 'metrics.txt')
        with open(f'results/rel/{vid}/masks/metrics.txt') as f:
            lines = f.readlines()
            if args.use_lpips and args.use_lpips:
                score_map, score_psnr, score_lpips, score_ssim = [float(s) for s in lines[2].split('\t')[:4]]
                scores.append([score_map, score_psnr, score_lpips, score_ssim])
            else:
                score_map, score_psnr = [float(s) for s in lines[2].split('\t')[:2]]
                scores.append([score_map, score_psnr])
    scores = np.array(scores).mean(axis=0)
    print('Average for all 10 scenes:')
    if args.use_lpips and args.use_lpips:
        print(f'mAP: {(scores[0] * 100).round(2)}, PSNR: {scores[1].round(2)}, LPIPS: {scores[2].round(2)}, SSIM: {scores[3].round(2)}')
    else:
        print(f'mAP: {(scores[0]*100).round(2)}, PSNR: {scores[1].round(2)}')


def render_video(args, model, dataset, root, save_cache=False):
    """Render a summary video like shown on the project page."""
    root = os.path.join(root, "summary")
    if os.path.exists(root):
        x = input('{} exists. Continue and replace?'.format(root))
    os.makedirs(root, exist_ok=True)

    sid = SAMPLE_IDS[args.vid]
    print('Rendering top row')
    top = evaluation.video.render(
        args, dataset, model, n_images=args.summary_n_samples,
    )
    print('Rendering bottom row, sid: {}'.format(sid))
    bot = evaluation.video.render(
        args, dataset, model, sid, n_images=args.summary_n_samples,
    )

    if save_cache:
        evaluation.video.save_to_cache(
            args.vid, sid, root=root, top=top, bot=bot
        )

    ims_cat = [
        evaluation.video.convert_rgb(
            evaluation.video.cat_sample(top[k], bot[k])
        )
        for k in bot.keys()
    ]

    save_name = f"{root}/cat-{sid}-N{len(ims_cat)}"
    utils.write_mp4(save_name, ims_cat)
    print('Video saved to {}'.format(save_name))

def run(args, model, dataset, root):

    if "masks" in args.outputs:
        # segmentations and renderings with mAP and PSNR
        eval_masks(args, model, dataset, root)

    if "summary" in args.outputs:
        # summary video
        render_video(args, model, dataset, root)


if __name__ == "__main__":
    args = parse_args()
    if 'average' in args.outputs:
        # calculate average over all 10 scenes for specific experiment
        eval_masks_average(args)
    else:
        model, dataset = init(args)
        root = os.path.join("results", args.exp, args.vid)
        run(args, model, dataset, root)
