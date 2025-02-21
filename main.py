
import numpy as np, argparse, time, random


from model import *

from trainer import train_or_eval_model

from dataloader import get_data_loaders
from transformers import AdamW



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')



if __name__ == '__main__':

    path = './saved_models/'  # 日志 模型保存路径

    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of gnn layers.')

    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')
    parser.add_argument('--dataset_name', default='MELD', type=str, help='dataset name, IEMOCAP, MELD, DailyDialog, EmoryNLP')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='learning rate')  #####
    parser.add_argument('--dropout', type=float, default=0.4, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help='batch size') ##


    parser.add_argument('--seed', type=int, default=100, help='random seed') ##

    parser.add_argument('--transformer_heads', type=int, default=4, help='Transformer encoder heads') 
    parser.add_argument('--transformer_layers', type=int, default=2, help='Transformer encoder layers')

    parser.add_argument('--weight_decay', type=float, default=0.01, 
                    help='Weight decay for AdamW optimizer.')

    parser.add_argument('--use_transformer', type=str2bool, default=True, 
                    help='Whether to use transformer module (ablation)')

    parser.add_argument('--use_cross_attention', type=str2bool, default=True,
                    help='Whether to use cross-attention between modules')



    args = parser.parse_args()

    print(args)

    # 固定随机种子
    seed_everything(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", args.device)

    device = args.device
    n_epochs = args.epochs
    batch_size = args.batch_size


    train_loader, valid_loader, test_loader = get_data_loaders(
        dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args=args)


    if 'IEMOCAP' in args.dataset_name:
        n_classes = 6
    else:
        n_classes = 7

    # Change this line:
    print('building model..')
    model = TripleGATs(args, n_classes)  # Changed from DualGATs
 
    if torch.cuda.device_count() > 1:
        print('Multi-GPU...........')
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    model.to(device)

    
    loss_function = nn.CrossEntropyLoss(ignore_index=-1) # 忽略掉label=-1 的类
    

    #optimizer = AdamW(model.parameters(), lr=args.lr)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay  # <--- This is where it's added
    )

    best_fscore, best_acc, best_loss, best_label, best_pred, best_mask = None, None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    best_acc = 0.
    best_fscore = 0.
    
    all_metrics = []  # Stores [valid_fscore, test_acc, test_fscore, test_loss]
    
    best_model = None
    for e in range(n_epochs):
        start_time = time.time()

        # Training phase
        train_loss, train_acc, _, _, train_fscore = train_or_eval_model(model, loss_function,
                                                                        train_loader, device,
                                                                        args, optimizer, True)
        # Validation phase
        valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_model(model, loss_function,
                                                                        valid_loader, device, args)
        # Test phase
        test_loss, test_acc, test_label, test_pred, test_fscore = train_or_eval_model(model, loss_function,
                                                                                      test_loader, device, args)

        # Store all metrics
        all_metrics.append([valid_fscore, test_acc, test_fscore, test_loss])

        print(
            'Epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(
                e + 1, 
                train_loss, train_acc, train_fscore,
                valid_loss, valid_acc, valid_fscore,
                test_loss, test_acc, test_fscore,
                round(time.time() - start_time, 2)
            ))

    print('\nfinish training!')

    # Sort by validation fscore (first element) descending
    all_metrics.sort(key=lambda x: x[0], reverse=True)

    # Best based on validation
    best_val_fscore, best_test_acc, best_test_fscore, best_test_loss = all_metrics[0]
    
    # Best overall test performance
    all_test_acc = [m[1] for m in all_metrics]
    all_test_fscore = [m[2] for m in all_metrics]
    all_test_loss = [m[3] for m in all_metrics]
    
    print('\n=== Final Results ===')
    print(f'Best Validation F1: {best_val_fscore:.2f}')
    print(f'Corresponding Test Metrics:')
    print(f' - Accuracy: {best_test_acc:.2f}%')
    print(f' - F1 Score: {best_test_fscore:.2f}%')
    print(f' - Loss: {best_test_loss:.4f}')
    
    print('\nBest Overall Test Performance:')
    print(f' - Max Accuracy: {max(all_test_acc):.2f}%')
    print(f' - Max F1 Score: {max(all_test_fscore):.2f}%')
    print(f' - Min Loss: {min(all_test_loss):.4f}')

