import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--device", type=str, default='cuda', help='target device, cuda or cpu')
    parser.add_argument("--log", type=str, help="log level", default='DEBUG')

    # datset configs
    parser.add_argument("--data_dir", type=str, default='dataset', help='dataset dir')
    parser.add_argument("--train_set", type=str, default='train.txt', help='txt file including the name of train houses')
    parser.add_argument("--test_set", type=str, default='test.txt', help='txt file including the name of test houses')

    # detection configs
    parser.add_argument("--det_config", type=str, default="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
            , help='default configs of the detection module (see Detectron2 Model Zoo for more details)')
    parser.add_argument("--det_model_weight", type=str, default='ckpt/model_final.pth', help='detection model')
    parser.add_argument("--det_save_images", action='store_true', help='save the detection outputs')
    parser.add_argument("--det_is_eval", action='store_true', help='only consider labeled images (for evaluation)')
    parser.add_argument("--det_check_labels", action='store_true', help='visualize random samples before training to check annotations')

    # layout estimation configs
    parser.add_argument("--lt_model_weight", type=str, default='ckpt/model_final.pth', help='layout estimation model')
    parser.add_argument('--lt_visualize', action='store_true')
    parser.add_argument('--lt_flip', action='store_true', help='whether to perfome left-right flip. ' '# of input x2.')
    parser.add_argument('--lt_rotate', nargs='*', default=[], type=float,
                        help='whether to perfome horizontal rotate. '
                             'each elements indicate fraction of image width. '
                             '# of input xlen(rotate).')
    parser.add_argument('--lt_r', default=0.05, type=float)
    parser.add_argument('--lt_min_v', default=None, type=float)
    parser.add_argument('--lt_force_cuboid', action='store_true')

    # room classification configs
    parser.add_argument("--rc_model", type=str, help="model architecture", default='unet')
    parser.add_argument("--rc_model_weight", type=str, help="weights file", default="src/exps/room_type_classification/room_type_final_model.pth")
    parser.add_argument("--rc_batch_size", type=int, help="batch_size", default=8)
    parser.add_argument("--rc_is_eval", action='store_true', help='only consider labeled images (for evaluation)')



    # arrangement configs
    parser.add_argument("--ar_model", type=str, help="model architecture", default='convmpn')
    parser.add_argument("--ar_exp", type=str, help="experiment name", default='test')
    parser.add_argument("--ar_batch_size", type=int, help="batch_size", default=1)
    parser.add_argument("--ar_model_weight", type=str, help="weights file", default="src/exps/main/model.pth")

    # visualization configs
    parser.add_argument('--vis_ignore_centers', action='store_true')
    parser.add_argument('--vis_ignore_door_colors', action='store_true')
    parser.add_argument('--vis_ignore_room_colors', action='store_true')

    # prediction configs
    parser.add_argument("--prediction_level", type=str, help="how many samples to use, lv1 is using 2 filters before prediction. For other values, also pass keep_sets_overlapped option (full, lv2, lv1)", default='lv1')


    # main configs
    parser.add_argument('--use_gt', action='store_true')
    parser.add_argument('--use_rotations_input', action='store_true')
    parser.add_argument('--keep_sets_overlapped', action='store_true')

    args = parser.parse_args()

    return parser
