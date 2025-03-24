import numpy as np

class Base:
    def __init__(self, base_dir, name, cats, files):
        
        self._base_dir = base_dir
        self.name = name
       
        cats, files = np.array(cats), np.array(files)
        
        self._image_files = files
        self._image_categories = cats

    
    @staticmethod
    def __name__(self):
        """
        Name of the  dataset
        """
        return self.name
        
    def __str__(self): 
        """
        Readable string representation
        """
        return "" + self.__name__() + "(" + str(self.__len__()) + ") in " + self.base_dir
    
    def __len__(self): 
        """
        Amount of elements
        """
        return len(self.image_files)

    @property
    def base_dir(self):
        """
        Path to the base directory
        
        Returns
        -------
        path : str
            Path to the base directory
        """
        return self._base_dir

    @property
    def image_files(self): 
        """
        List of image files. The order of the list is important for other methods.
        
        Returns
        -------
        file_list : list(str)
            List of file names
        """
        return self._image_files
    
    @property
    def image_categories(self): 
        """
        List of image_categories.
        
        Returns
        -------
        file_list : list(str)
            List of image_categories
        """
        return self._image_categories