import numpy as np

class Parser():
    '''    Pierwsze -> 0
           Złożone  -> 1  '''
    def __init__(self, data_filename, labels_filename):
        self.width  = 0
        self.height = 0
        
        data = self.parse_files(data_filename, labels_filename)
        values = list(np.where(np.array(data[1]) == 1)[1])
        data = list(filter(lambda x : x[2] not in [0, 1], list(zip(data[0], data[1], values))))
        self.size = len(data)
        self.data   = [ [data[i][0], 0] if data[i][2] in [2, 3, 5, 7] else \
                        [data[i][0], 1] for i in range(self.size)]


    def parse_files(self, image_filename, labels_filename):
        with open(image_filename, 'rb') as f:
            file_bytes = f.read()

        with open(labels_filename, 'rb') as l:
            labels_bytes = l.read()

        length = int.from_bytes(file_bytes[ 4: 8], byteorder='big')
        self.height = int.from_bytes(file_bytes[ 8:12], byteorder='big')
        self.width  = int.from_bytes(file_bytes[12:16], byteorder='big')
        self.size   = length

        index_im = 16
        index_lb = 8
        images = []
        labels = []

        for i in range(length):
            temp = np.zeros( (self.width*self.height), dtype = np.float )
            for y in range(self.height):
                for x in range(self.width):
                    temp[y*self.width+x] = file_bytes[index_im]/255.0
                    index_im += 1

            images.append(temp)

            label = [ 1 if x == labels_bytes[index_lb] else 0 for x in range(10)]
            labels.append(label)
            index_lb += 1

        return images, labels
