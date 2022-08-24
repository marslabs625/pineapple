import time
import matplotlib.pyplot as plt

class Timer():
    '''
    loop timer
    '''
    def __init__(self, total_loops=None):
        self.total_loops = total_loops
        self.times = 0
        
    def start(self):
        self.start_time = time.perf_counter()
        self.times += 1
        
    def finish(self):
        self.elapsed_time = time.perf_counter() - self.start_time
        
        if self.total_loops != None:
            self.ETA = self.elapsed_time * (self.total_loops - self.times)
            self.ETA = time.strftime('%H:%M:%S', time.gmtime(self.ETA))
        
    def reset_times(self):
        self.times = 0
        
class Plotter():
    '''
    templates for printing
    '''
    def __init__(self):
        self.data_list = []
        
    def append_data(self, data, fmt, label):
        data.append(fmt)
        data.append(label)
        self.data_list.append(data)
        
    def reset_dataList(self):
        self.data_list = []
        
    def line_chart(self, savepath, x_label, y_label, x_min, x_max, x_interval):
        plt.figure()
        
        for data in self.data_list:
            label = data.pop(-1)
            fmt = data.pop(-1)
            plt.plot(data, fmt, label=label)
            
        plt.ylabel(x_label)
        plt.xlabel(y_label)
        plt.xticks([x for x in range(x_min, x_max, x_interval)])
        plt.legend()
        plt.savefig(savepath)