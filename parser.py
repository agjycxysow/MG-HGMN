"""Getting params from the command line."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run model.")

    # General args:
    parser.add_argument("--model-name",type=str,
                        default='tryout',help="model name, Default = [SimGNN, GMN, slotGAT, myGAT(SimpleHGN/HGB)]")
    parser.add_argument("--dataset",type=str,
                        default='ACMNEW',help="dataset name")
    parser.add_argument("--server", type=str,
                        default='3')

    parser.add_argument("--result-path",type=str,
                        default='/home/jiaozihao/Projects/demo2/result/',help="Where to save the evaluation results")
    parser.add_argument("--model-train", type=int,
                        default=1, help='Whether to train the model')
    # 1: train; 0: test.
    parser.add_argument("--model-path-4", type=str,
                        default=f'/home/jiaozihao/Projects/demo2/models/', help="Where to save the trained model")
    parser.add_argument("--model-path-3", type=str,
                        default=f'/home/jzh/Projects/demo2/models/', help="Where to save the trained model")

    parser.add_argument("--graph-pair-mode", type=str,
                        default='combine',
                        help="The way of generating graph pairs, including [normal, delta, combine].")
    parser.add_argument("--target-mode", type=str,
                        default='exp', help="The way of generating target, including [linear, exp].")
    parser.add_argument("--num-testing-graphs", type=int,
                        default=200, help="The number of testing graph pairs for each graph. 1/5 of total graphs ")


    # Model general args:
    parser.add_argument("--gpu", type=bool,
                        default=False,help="whether to use gpu train model.")
    parser.add_argument("--wandb", type=bool,
                        default=False, help="whether to log running.")
    parser.add_argument("--whole", type=bool,
                        default=True, help="whether to use full ACM dataset training.")

    parser.add_argument("--homo",type=bool,
                        default=True,help="whether run homogeneous model, default=True")
    parser.add_argument("--epochs", type=int,
                        default=1, help="Number of training epochs. Default is 1(GEDGNN)/5(SimGNN).")
    parser.add_argument("--model-epoch-start", type=int,
                        default=0, help="The number of epochs the initial saved model has been trained.")
    parser.add_argument("--model-epoch-end", type=int,
                        default=20, help="The number of epochs the final saved model has been trained.""default = 20")



    # Model hyperparameters:
    parser.add_argument("--ln", type=bool,
                        default=True, help="whether use LayerNorm.")
    parser.add_argument("--batch-size",type=int,
                        default=128,help="Number of graph pairs per batch. Default is 128.")
    parser.add_argument("--learning-rate",type=float,
                        default=5e-4,   help="Learning rate. Default is 0.001(simgnn).")
    parser.add_argument("--lr_reduce_factor",
                        default=0.5)
    parser.add_argument("--lr_schedule_patience",
                        default=20)
    parser.add_argument("--min_lr",
                        default=1e-6)
    parser.add_argument("--momentum", type=float,
                        default=0.9, help="SGD momentum. Default is 0.9.")
    parser.add_argument("--dropout", type=float,
                        default=0.5, help="Dropout probability. Default is 0.5.")
    parser.add_argument("--weight-decay",type=float,
                        default=5e-4,help="Adam weight decay. Default is 5*10^-4.")
    parser.add_argument("--loss-weight",type=float,
                        default=1.0,help="In GedGNN, the weight of value loss. Default is 1.0.")

    # Homogeneous model args:
    parser.add_argument("--homo-input", type=str,
                        default="original", help="For homogeneous model, use what input features, "
                                                 "[original, typehot, onehot]")
    parser.add_argument("--bins",type=int,default=16)
    parser.add_argument("--histogram",dest="histogram",
                        default=True,help='Whether to use histogram.')
    parser.add_argument("--filters-1",type=int,
                        default=128,help="Filters (neurons) in 1st convolution. Default is 128.")
    parser.add_argument("--filters-2",type=int,
                        default=64,help="Filters (neurons) in 2nd convolution. Default is 64.")
    parser.add_argument("--filters-3",type=int,
                        default=32,help="Filters (neurons) in 3rd convolution. Default is 32.")
    parser.add_argument("--num-layers",type=int,
                        default=3,help="slot layer number. Default is 3.")
    parser.add_argument("--tensor-neurons",type=int,
                        default=16,help="Neurons in tensor network layer. Default is 16.")
    parser.add_argument("--bottle-neck-neurons",type=int,
                        default=16,help="Bottle neck layer neurons. Default is 16.")
    parser.add_argument("--bottle-neck-neurons-2",type=int,
                        default=8,help="2nd bottle neck layer neurons. Default is 8.")
    parser.add_argument("--bottle-neck-neurons-3",type=int,
                        default=4,help="3rd bottle neck layer neurons. Default is 4.")

    # GMN args:
    parser.add_argument("--gcn-size", type=int, default=[128, 64, 32], nargs='+')
    parser.add_argument("--similarity", type=str, default="cosine")
    parser.add_argument("--readout", type=str, default="gated")


    # SlotGAT args:
    parser.add_argument("--edge-dim",type=int,
                        default=32,help="Edge features dim passed in slot. Default is 32.")
    parser.add_argument("--alpha",type=float,
                        default=0.05,help="slot gat alpha. Default is 0.05.")
    parser.add_argument("--slope",type=float,
                        default=0.05,help="slot gat slope. Default is 0.05.")
    parser.add_argument("--hidden-dim",type=int,
                        default=64,help="slot hidden dim. Default is 64.")
    parser.add_argument("--num-classes",type=int,
                        default=8,help="slot output dim (Node Classification). Default is 8.")
    parser.add_argument("--pad",type=int,
                        default=128,help="No attribute node type padding vector length")

    # channel args:
    parser.add_argument("--channel-1", type=int,
                        default=64, help="channel embedding 1.")




    parser.add_argument("--demo",
                        dest="demo",
                        action="store_true",
                        default=False,
                        help='Generate just a few graph pairs for training and testing.')

    parser.add_argument("--gtmap",
                        dest="gtmap",
                        action="store_true",
                        default=False,
                        help='Whether to pack gt mapping')

    parser.add_argument("--save-path",
                        type=str,
                        default=None,
                        help="Where to save the trained model")

    parser.add_argument("--load-path",
                        type=str,
                        default=None,
                        help="Load a pretrained model")


    return parser.parse_args()
