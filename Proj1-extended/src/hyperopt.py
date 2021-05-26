import click

from test import run


@click.command()
@click.option('--model', default='ConvNet',
              type=click.Choice(['ConvNet', 'MLP'], case_sensitive=False),
              help="Model to evaluate.")
@click.option('--siamese/--no-siamese', default=False, type=bool,
              help="Use a siamese version of the model.")
@click.option('--outputfile', default='hyperparameters.csv', type=str,
              help="File in which to store tuning results.")
@click.option('--verbose', default=1, type=int,
              help="Print out info for debugging purposes.")
@click.pass_context
def tune_hyperparams(ctx: click.Context, model: str, siamese: bool = False,
                     outputfile: str = 'hyperparameters.csv', verbose: int = 0) -> None:
    
    learning_rates = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    weight_decays = [1e-4, 1e-3, 1e-2, 1e-1]
    
    auxiliary_contributions = [0., 0.25, 0.5, 0.75, 1.] if siamese else [0.]
    
    if verbose > 0:
        print(f"Saving performance results in {outputfile}...")
    
    outputfile = model + outputfile
    if siamese:
        outputfile = "Siamese" + outputfile
    
    # Header in output file
    with open(outputfile, 'w+') as outfile:
        outfile.write('learning_rate,decay,gamma,train_accuracy,test_accuracy\n')
          
    for lr in learning_rates:
        for decay in weight_decays:
            for gamma in auxiliary_contributions:
                
                if verbose > 0:
                    print(f"\nComputing performance for learning rate {lr}, "
                          f"decay {decay}, and gamma {gamma}")
                
                train_accuracy, test_accuracy = \
                    ctx.invoke(run, model=model, siamese=siamese, epochs=25,
                               lr=lr, decay=decay, gamma=gamma,
                               trials=3, seed=123456,
                               batch_size=50, standardize=True,
                               make_figs=False, clear_figs=False,
                               verbose=verbose)
                
                if verbose > 0:
                    print(f"Train {train_accuracy}, Test {test_accuracy}.")
                        
                with open(outputfile, 'a') as outfile:
                    outfile.write(f"{lr},{decay},{gamma},{train_accuracy},{test_accuracy}\n")
        

if __name__ == '__main__':
    tune_hyperparams()
