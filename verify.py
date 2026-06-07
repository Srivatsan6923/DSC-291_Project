import os
import torch
import argparse
import pandas as pd
from utils import get_dataloaders
from verification import verify, verified_acc_bruteforce, verified_acc_ibp, attack_charmer, verify_rsdel

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of a shallow transformer')
    parser.add_argument('--dataset', default='ag_news', type=str, help='glue datase or numbers_letters')
    parser.add_argument('--n_classes', default=4, type=int, help='number of classes')
    parser.add_argument('--model_path', default='results/example_ag_news/weights_last.pt', type=str, help='default model weights')
    parser.add_argument('--n_samples', default=-1, type=int, help='number of samples to consider')
    parser.add_argument('--suffix', default='', type=str,
                    help='suffix added to output csv names')
    parser.add_argument(
    '--method',
    default='all',
    choices=['all', 'ours', 'lipslev', 'charmer', 'bf', 'ibp', 'rsdel'],
    help='verification method'
)

    args = parser.parse_args()

    

    if args.dataset in ['ag_news', 'imdb', 'fake-news']:
        pad = 1000
        pad_verif = 1010
    else:
        pad = 286
        pad_verif = 296

    out_folder = os.path.abspath(os.path.join(args.model_path, os.pardir))
    print(out_folder)

    char_to_id, train_loader, valid_loader, test_loader = get_dataloaders(args.dataset,16,pad = pad, valid_size=0)
    # create the model.
    n_char=len(char_to_id.keys())

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Using GPU?', cuda)

    if 'sentence' in train_loader.dataset[0]:
        key = 'sentence'
    else:
        key = 'text'
    
    net = torch.load(args.model_path, map_location=device, weights_only=False)

    if hasattr(net, "char_to_id"):
        char_to_id = net.char_to_id

    if not hasattr(net,'lips_act'):
        net.lips_act = 1

    if not hasattr(net,'lips_emb'):
        net.lips_emb = True
        net.batch_norm = False

    if args.dataset != 'numbers_letters':
        if args.n_samples == -1:
            l = len(test_loader.dataset)
        else:
            l = min(args.n_samples, len(test_loader.dataset))

        cut_dataset = [
            {
                key: str(test_loader.dataset[i][key])[:1000],
                'label': test_loader.dataset[i]['label']
            }
            for i in range(l)
        ]
    else:
        char_to_id = net.char_to_id
        cut_dataset = test_loader.dataset

    output_name = args.suffix

    if args.method in ['all', 'bf']:
        path = os.path.join(out_folder, f'{output_name}_results_bruteforce.csv')
        if os.path.isfile(path):
            df = pd.read_csv(path)
            if len(df) < args.n_samples:
                print('bf', verified_acc_bruteforce(net, char_to_id, cut_dataset, device, output_folder=out_folder, output_name=output_name, pad=pad_verif, resume=True))
        else:
            print('bf', verified_acc_bruteforce(net, char_to_id, cut_dataset, device, output_folder=out_folder, output_name=output_name, pad=pad_verif))

    if args.method in ['all', 'ibp']:
        path = os.path.join(out_folder, f'{output_name}_results_ibp.csv')
        if os.path.isfile(path):
            df = pd.read_csv(path)
            if len(df) < args.n_samples:
                print('ibp', verified_acc_ibp(net, char_to_id, cut_dataset, device, output_folder=out_folder, output_name=output_name, pad=pad_verif, resume=True))
        else:
            print('ibp', verified_acc_ibp(net, char_to_id, cut_dataset, device, output_folder=out_folder, output_name=output_name, pad=pad_verif))

    if args.method in ['all', 'charmer']:
        path = os.path.join(out_folder, f'{output_name}_results_charmer.csv')
        if os.path.isfile(path):
            df = pd.read_csv(path)
            if len(df) < args.n_samples:
                print('charmer', attack_charmer(net, char_to_id, cut_dataset, device, output_folder=out_folder, output_name=output_name, pad=pad_verif, resume=True))
        else:
            print('charmer', attack_charmer(net, char_to_id, cut_dataset, device, output_folder=out_folder, output_name=output_name, pad=pad_verif))

    if args.method in ['all', 'rsdel']:
        path = os.path.join(out_folder, f'{output_name}_results_rsdel.csv')
        if os.path.isfile(path):
            df = pd.read_csv(path)
            if len(df) < args.n_samples:
                print('rsdel', verify_rsdel(net, char_to_id, cut_dataset, device, output_folder=out_folder, output_name=output_name, pad=pad_verif, resume=True))
        else:
            print('rsdel', verify_rsdel(net, char_to_id, cut_dataset, device, output_folder=out_folder, output_name=output_name, pad=pad_verif))

    if args.method in ['all', 'ours', 'lipslev']:
        path = os.path.join(out_folder, f'{output_name}_results_lipslev.csv')
        if os.path.isfile(path) and len(pd.read_csv(path)) >= len(cut_dataset):
            print('ours skipped existing', path)
        else:
            print('ours', verify(net, char_to_id, cut_dataset, device, output_folder=out_folder, output_name=output_name, pad=pad_verif))
