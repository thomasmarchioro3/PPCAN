import csv
import os

class measure_container:
    def __init__(self, aim="high", MAX_VALUE=1000):
        assert (aim=="high" or aim=="low")
        self.e_matrix = []
        self.e_list = []
        self.aim=aim
        self.MAX_VALUE = MAX_VALUE
        self.e_best = 0 if self.aim=="high" else self.MAX_VALUE
        self.i_best = 0
        self.overall_best = 0 if self.aim=="high" else self.MAX_VALUE
        self.iter_list = []
        self.iter_start = 0
        return
    
    def initialize_list(self):
        self.e_best = 0 if self.aim=="high" else self.MAX_VALUE
        self.i_best = 0
        self.e_list = []
        return
    
    def get_list(self):
        return self.e_list
    
    def get_last_elem(self):
        assert self.e_list
        return self.e_list[-1]
    
    def get_best_elem(self):
        return self.e_best
    
    def append_elem(self, e):
        assert e <= self.MAX_VALUE
        if(self.aim == "high"):
            if(e>self.e_best):
                self.e_best=e
                self.i_best=len(self.e_list)
            if(e>self.overall_best):
                self.overall_best=e
        else: # self.aim == "low"
            if(e<self.e_best):
                self.e_best=e
                self.i_best=len(self.e_list)
            if(e<self.overall_best):
                self.overall_best=e
        self.e_list.append(e)
        return
    
    def initialize_matrix(self):
        self.overall_best = 0 if self.aim=="high" else self.MAX_VALUE
        self.e_matrix = []
        return
    
    def append_list(self):
        if(self.e_matrix):
            assert len(self.e_matrix[-1]) == len(self.e_list)
        self.e_matrix.append(self.e_list)
        return
    
    def get_matrix(self):
        return self.e_matrix
    
    def get_last_col(self):
        assert self.e_matrix
        return self.e_matrix[-1]
    
    def get_overall_best(self):
        return self.overall_best

    def initialize_iter_list(self):
        self.iter_list = []
        return
    
    def set_iter_start(self, iter_start):
        self.iter_start = iter_start
        return

    def get_current_iter(self):
        if(not self.iter_list):
            return self.iter_start
        else:
            return self.iter_list[-1]

    def update_iter_list(self, num_iters):
        self.iter_list.append(self.get_current_iter()+num_iters)
        return

    def get_iter_list(self):
        return self.iter_list
    
    def write_matrix(self, filename="results.csv", directory="results"):
        assert len(self.iter_list) == len(self.e_matrix[-1])
        if directory != "" and not os.path.exists(directory):
            os.makedirs(directory)
        filename = filename if directory=="" else directory+"/"+filename
        with open(filename, 'w+') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=' ')
            for i in range(len(self.e_matrix[-1])):
                row = []
                row.append(self.iter_list[i])
                for j in range(len(self.e_matrix)):
                    row.append(self.e_matrix[j][i])
                csv_writer.writerow(row)