"""
The Association class represents an association between two concepts, which are represented 
by their respective vectors. This class can be used to store and manage associations between different concepts.

When an instance of the Association class is created, it takes:
    - Two vectors (vector1 and vector2) 
    - A description of the association between them (association_description). 

These associations will be stored in a Firebase database and accessed by the associative memory agent.

"""
class Association:
    def __init__(self, vector1, vector2, association_description):
        """
        Initializes an instance of the Association class with the specified vectors and description.

        Args:
            vector1 (list): The first vector in the association.
            vector2 (list): The second vector in the association.
            association_description (str): A description of the association.
        """
        self.vector1 = vector1
        self.vector2 = vector2
        self.association_description = association_description
        
    def to_firebase(self):
        """
        Converts the Association instance to a dictionary that can be stored in a Firebase database.

        Returns:
            dict: A dictionary containing the association's fields as key-value pairs.
        """
        return {
            "vector1": self.vector1,
            "vector2": self.vector2,
            "association_description": self.association_description
        }