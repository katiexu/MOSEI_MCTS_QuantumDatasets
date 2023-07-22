class Arguments:
    def __init__(self):
        self.n_qubits   = 12
        self.device     = 'cpu'
        self.clr        = 0.005
        self.qlr        = 0.05
        self.epochs     = 5
        self.batch_size = 128
        self.test_batch_size = 1500
        self.ntrials    = 50