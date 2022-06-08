'''
Evaluate universal adversarial attack.

Evaluate by counting average number of edits between original input
(with attack phrase) and  GEC model prediction

Also:
- Report how many edits average in original part of sentence and adversarial part.
- Plot histogram of error distibution before and after attack
- Show errant error type distribution before and after attack 
'''

import sys
import os
import argparse
from utils.helpers import read_lines, normalize
from gector.gec_model import GecBERTModel


def count_edits(input_file, model, attack_phrase=''):
    test_data = read_lines(input_file)
    cnt_corrections = 0
    num_0_edits = 0
    for i,sent in enumerate(test_data):
        print(f'On {i}/{len(test_data)}')
        sent_attack = sent + ' ' + attack_phrase + ' .'
        _, cnt = model.handle_batch([sent_attack.split()])
        cnt_corrections += cnt

    return cnt_corrections
    #return cnt_corrections/len(test_data)


def main(args):
    # load model
    model = GecBERTModel(vocab_path=args.vocab_path,
                            model_paths=args.model_path,
                            max_len=args.max_len, min_len=args.min_len,
                            iterations=args.iteration_count,
                            min_error_probability=args.min_error_probability,
                            lowercase_tokens=args.lowercase_tokens,
                            model_name=args.transformer_model,
                            special_tokens_fix=args.special_tokens_fix,
                            log=False,
                            confidence=args.additional_confidence,
                            del_confidence=args.additional_del_confidence,
                            is_ensemble=args.is_ensemble,
                            weigths=args.weights)
    
    avg_edits, frac_0_edits = count_edits(args.input_file, model, attack_phrase = args.attack_phrase) 
    print(f"Average Edits {avg_edits}" )



if __name__ == "__main__":

    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        required=True)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
                                 'bert-large', 'roberta-large', 'xlnet-large'],
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--additional_del_confidence',
                        type=float,
                        help='How many probability to add to $DELETE token.',
                        default=0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='Minimum probability for each action to apply. '
                             'Also, minimum error probability, as described in the paper.',
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--attack_phrase',
                        type=str,
                        help='attack phrase to concatenate', nargs='+',
                        default='')
    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_uni_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n') 
    
    main(args)







