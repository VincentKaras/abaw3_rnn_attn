from cli.parsers import Options
from training.training_process import TrainingProcess

if __name__ == "__main__":
    
    params = Options().parse()
    
    trainer = TrainingProcess(params)
    
    trainer.start()