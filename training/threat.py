
class Threat:
    def __init__(self, data):
        self.name = None
        self.id = None
        self.path = None

        if "_id" in data:
            self.id = data["_id"]
        else:
            raise ValueError("Threat must have an _id")
        
        if "name" in data:
            self.name = data["name"]
        else:
            raise ValueError("Threat must have a name")
        
        if "path" in data:
            self.path = data["path"]
        else:
            raise ValueError("Threat must have a path")
