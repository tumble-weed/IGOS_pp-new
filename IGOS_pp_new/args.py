import argparse


def init_args():

    parser = argparse.ArgumentParser(
        description='Generate explanations using I-GOS and iGOS++.'
    )

   
    parser.add_argument(
        '--input_size',
        type=int,
        default=224,
        help='The input size to the network.')


    parser.add_argument(
        '--method',
        required=True,
        type=str,
        choices=['I-GOS', 'iGOS+', 'iGOS++'],
        default='I-GOS'
    )

    parser.add_argument(
        '--opt',
        required=True,
        type=str,
        choices=['LS', 'NAG'],
        default='NAG',
        help='The optimization algorithm.'
    )

    parser.add_argument(
        '--diverse_k',
        type=int,
        default=2)

    parser.add_argument(
        '--init_posi',
        type=int,
        default=0,
        help='The initialization position, which cell of the K x K grid will be used to initialize the mask with nonzero values (use init_val to control it)')
    """
            If K = 2:      If K = 3:
            -------        ----------
            |0 |1 |        |0 |1 |2 |
            -------        ----------
            |2 |3 |        |3 |4 |5 |
            -------        ----------
                           |6 |7 |8 |
                           ----------
    """

    parser.add_argument(
        '--init_val',
        type=float,
        default=0.,
        help='The initialization value used to initialize the mask in only one cell of the K x K grid.')

    parser.add_argument(
        '--L1',
        type=float,
        default=1
    )

    parser.add_argument(
        '--L2',
        type=float,
        default=20
    )

    parser.add_argument(
        '--ig_iter',
        type=int,
        default=20)

    parser.add_argument(
        '--iterations',
        type=int,
        default=15
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=1000
    )

    return parser.parse_args()
